"""Video encoding via ffmpeg.

Two frame rates live here and must not be confused. The *hold* rate is how fast
photographs advance -- one per week at 0.5/s means two seconds each. The
*playback* rate is the video's own, and only differs from the hold rate when a
transition is asked for, because a dissolve exists precisely in the frames
between two photographs.

Everything that decides a filter string is plain arithmetic in
`transition_filters`, so it stays testable in an environment with no ffmpeg --
which is every environment the test suite runs in.

One render uses one core, and no flag changes that. `minterpolate` does not
declare slice threading in libavfilter, so `-threads`, `-filter_threads` and
`-filter_complex_threads` have nothing to distribute -- they only spread work
across filters that opted in. x264 underneath is threaded but spends a morph
starved, waiting on a filter that emits one frame at a time. More cores
therefore means more ffmpeg processes, which is why `pipeline` runs the plain
and annotated videos at once rather than one after the other.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

TRANSITIONS = ("none", "crossfade", "morph")


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegMissing(
            "ffmpeg not found on PATH. Install it (brew install ffmpeg / "
            "apt install ffmpeg) and re-run."
        )
    return path


def fingerprint(frames: list[tuple[str, str, str]], settings: dict,
                extra: dict | None = None) -> str:
    """A digest of everything that decides the bytes ffmpeg will write.

    `frames` is one `(asset_id, path, warped_at)` per photograph, in order. All
    three matter and for different reasons: the ids and their order are the
    video's content, the paths are where the pixels are read from, and
    `warped_at` is `align`'s own record of when it last rewrote each frame --
    which is what catches a re-align that changed the framing without changing
    which photographs were chosen. Comparing file contents would be the other
    way to notice that, at the cost of reading every frame to decide whether to
    read every frame.

    Pure, and JSON rather than `hash()`, because the digest is written to the
    manifest and has to mean the same thing in the next process. `sort_keys`
    for the same reason: dict order must not decide whether a render is reused.
    """
    payload = json.dumps(
        {"frames": [list(f) for f in frames],
         "settings": {k: settings[k] for k in sorted(settings)},
         "extra": extra or {}},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def moving_seconds(hold_fps: float, transition_seconds: float) -> float:
    """How much of one photo's slot is spent in motion, clamped to the slot.

    A transition longer than the photo it leads out of cannot be placed at all;
    clamping yields a picture permanently in motion, which is a real look rather
    than an error.
    """
    hold = 1.0 / float(hold_fps)
    return max(0.0, min(float(transition_seconds), hold))


def concat_entries(frames: list[Path], hold_fps: float, transition: str,
                   transition_seconds: float) -> list[tuple[Path, float]]:
    """The (file, duration) pairs an ffconcat list should contain.

    `crossfade` and `none` get one entry per photograph: the dissolve is carved
    out of the gap by `framerate`'s interp window, so the input needs no help.

    `morph` gets two, the same picture twice -- held for the still part of the
    slot, then again for the moving part. `minterpolate` has no notion of a
    hold; it spreads synthesised frames evenly across every gap between input
    frames, so left alone the picture never stands still. Feeding it a
    duplicated frame gives it a gap with no motion in it to find, and confines
    the morph to the gap that follows.
    """
    hold = 1.0 / float(hold_fps)
    if transition != "morph":
        return [(frame, hold) for frame in frames]

    moving = moving_seconds(hold_fps, transition_seconds)
    still = hold - moving
    if still <= 0:
        # Nothing to hold: the morph fills the slot, which is what it did before
        # it had a hold at all. Emitting a zero-length entry instead would put a
        # `duration 0.000000` in front of ffmpeg for no reason.
        return [(frame, hold) for frame in frames]
    return [pair for frame in frames for pair in ((frame, still), (frame, moving))]


def write_entries(entries: list[tuple[Path, float]], list_path: Path) -> Path:
    """Write prepared (file, duration) pairs as an ffconcat list."""
    if not entries:
        raise ValueError("no frames to encode")

    lines = ["ffconcat version 1.0"]
    for frame, held in entries:
        lines.append(f"file '{frame.resolve().as_posix()}'")
        lines.append(f"duration {held:.6f}")
    # ffmpeg drops the final entry's duration unless the file is repeated, which
    # otherwise makes the last frame flash past in a single tick.
    lines.append(f"file '{entries[-1][0].resolve().as_posix()}'")

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def write_concat_list(frames: list[Path], fps: float, list_path: Path,
                      first_duration: float | None = None) -> Path:
    """Write an ffconcat list holding every frame for the same duration.

    The concat demuxer is used rather than a `frame_%06d.png` glob so that
    manually rejected frames can simply be omitted -- no renumbering of files on
    disk, which would invalidate the manifest.

    `first_duration` shortens only the opening entry, which shifts every later
    one earlier by the difference. The footer overlay needs exactly that: see
    `footer_concat_offset`.
    """
    if not frames:
        raise ValueError("no frames to encode")

    duration = 1.0 / float(fps)
    entries = [(frame, duration if index or first_duration is None
                else float(first_duration))
               for index, frame in enumerate(frames)]
    return write_entries(entries, list_path)


def dissolve_bounds(fraction: float) -> tuple[int, int]:
    """`framerate`'s interp window for a dissolve covering `fraction` of a hold.

    The filter walks 0..255 across the gap between two source frames and blends
    only between these two marks, holding the nearer frame outside them. Centring
    the window on 128 puts the dissolve on the boundary between two photographs,
    with equal stillness either side.
    """
    span = 127.5 * max(0.0, min(1.0, fraction))
    return round(127.5 - span), round(127.5 + span)


def footer_concat_offset(hold_fps: float, transition: str = "crossfade",
                         transition_seconds: float = 0.0) -> float:
    """How long the *first* footer holds, so switches land mid-transition.

    Photographs carry their own timestamps, which fall at the *end* of the
    transition leading into them -- a footer following them directly would leave
    the next photo on screen already, still labelled with the previous date.
    Opening short moves every switch back to the moment the picture stops being
    mostly one photograph and starts being mostly the next.

    Where that moment falls depends on the transition. A crossfade straddles the
    boundary, so it is the middle of the slot. A morph sits at the *end* of the
    slot -- a duplicated frame can only hold from the start of one -- so its
    midpoint is half a morph before the boundary instead.
    """
    hold = 1.0 / float(hold_fps)
    if transition == "morph":
        return hold - moving_seconds(hold_fps, transition_seconds) / 2
    return hold / 2


def transition_filters(transition: str, hold_fps: float, playback_fps: float,
                       transition_seconds: float) -> tuple[list[str], float]:
    """The filters for a transition, and the frame rate the output must carry.

    Returning the rate is not decoration. `minterpolate` and `framerate` both
    raise the frame rate inside the filter graph, and an output `-r` left at the
    hold rate throws every synthesised frame away again -- the expensive work
    happens and nothing reaches the file.
    """
    if transition not in TRANSITIONS:
        raise ValueError(
            f"unknown transition {transition!r}; expected one of {TRANSITIONS}")
    if transition == "none":
        return [], float(hold_fps)

    if transition == "morph":
        # scd=none for the same reason as scene=100 below. minterpolate carries
        # its own scene-change detector, on by default, and when it fires it
        # stops interpolating and duplicates frames -- so a timelapse, where
        # every consecutive pair is a huge difference, pays for motion
        # compensation and gets hard cuts back. The morph's timing is not here:
        # minterpolate has no hold, so it is built into the input by
        # `concat_entries`.
        return ([f"minterpolate=fps={playback_fps:g}:mi_mode=mci:mc_mode=aobmc"
                 ":me_mode=bidir:vsbmc=1:scd=none"], float(playback_fps))

    # A dissolve longer than the hold cannot be centred on anything; clamping
    # yields a continuous blend, which is a real look rather than an error.
    start, end = dissolve_bounds(float(transition_seconds) * float(hold_fps))
    # scene=100 disables scene-change detection. Its default of 8.2 treats
    # consecutive photographs months apart as a cut and declines to blend them,
    # which is every pair in a timelapse -- the filter would run, cost its time,
    # and hand back the hard cuts it was added to remove.
    return ([f"framerate=fps={playback_fps:g}:interp_start={start}"
             f":interp_end={end}:scene=100"], float(playback_fps))


def build_command(list_path: Path, out_path: Path, fps: float, codec: str,
                  crf: int, extra_filters: list[str] | None = None,
                  output_fps: float | None = None,
                  overlay_list: Path | None = None) -> list[str]:
    filters = list(extra_filters or [])
    # yuv420p requires even dimensions; odd output geometry otherwise fails at
    # the very last step, after all the expensive work.
    guard = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    rate = float(output_fps) if output_fps is not None else float(fps)

    cmd = [ffmpeg_binary(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
    if overlay_list is None:
        cmd += ["-vf", ",".join([*filters, guard])]
    else:
        # The footer rides a second input so the transition cannot touch it: its
        # stream is resampled with `fps`, which repeats frames and never blends,
        # and is composited only after the pictures have finished dissolving.
        # format=rgba is explicit because losing the alpha shows up as a black
        # band behind the text rather than as an error.
        picture = ",".join(filters) if filters else "null"
        cmd += [
            "-f", "concat", "-safe", "0", "-i", str(overlay_list),
            "-filter_complex",
            f"[0:v]{picture}[img];"
            f"[1:v]fps={rate:g},format=rgba[ftr];"
            # The footer stream is half a hold shorter than the picture, because
            # its first entry is clipped to move the switches onto the dissolve
            # midpoints. eof_action=repeat holds the last footer over that tail
            # rather than dropping it; it is the default, said out loud because
            # the alternative silently unlabels the final photograph.
            f"[img][ftr]overlay=0:0:eof_action=repeat[ov];"
            f"[ov]{guard}[v]",
            "-map", "[v]",
        ]
    cmd += [
        # str() rather than a :g format so a transition-free command stays
        # character-for-character what it has always been.
        "-r", str(rate),
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
    ]
    if codec in ("libx264", "libx265"):
        cmd += ["-crf", str(crf), "-preset", "slow"]
    else:
        # videotoolbox and friends have no CRF; drive quality instead.
        cmd += ["-q:v", "50"]
    cmd.append(str(out_path))
    return cmd


def encode(frames: list[Path], out_path: Path, fps: float = 10.0,
           codec: str = "libx264", crf: int = 18,
           interpolate: bool = False, transition: str = "none",
           playback_fps: float = 30.0, transition_seconds: float = 1.0,
           overlays: list[Path] | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `interpolate` predates `transition` and named the same idea, so it still
    # selects it. An explicit transition wins; only a config written before the
    # setting existed can reach the fallback.
    if interpolate and transition == "none":
        transition = "morph"

    filters, rate = transition_filters(transition, fps, playback_fps, transition_seconds)

    stem = out_path.stem
    list_path = write_entries(
        concat_entries(frames, fps, transition, transition_seconds),
        out_path.parent / f"{stem}.ffconcat")
    overlay_list = None
    if overlays and transition != "none":
        # The footer is never doubled: it holds and switches, so one entry each
        # is right whatever the picture underneath is doing.
        overlay_list = write_concat_list(
            overlays, fps, out_path.parent / f"{stem}-footer.ffconcat",
            first_duration=footer_concat_offset(fps, transition, transition_seconds))

    cmd = build_command(list_path, out_path, fps, codec, crf, filters,
                        output_fps=rate, overlay_list=overlay_list)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{tail}")
    return out_path
