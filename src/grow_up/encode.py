"""Video encoding via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


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


def write_concat_list(frames: list[Path], fps: float, list_path: Path) -> Path:
    """Write an ffconcat list.

    The concat demuxer is used rather than a `frame_%06d.png` glob so that
    manually rejected frames can simply be omitted -- no renumbering of files on
    disk, which would invalidate the manifest.
    """
    if not frames:
        raise ValueError("no frames to encode")

    duration = 1.0 / float(fps)
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        lines.append(f"file '{frame.resolve().as_posix()}'")
        lines.append(f"duration {duration:.6f}")
    # ffmpeg drops the final entry's duration unless the file is repeated, which
    # otherwise makes the last frame flash past in a single tick.
    lines.append(f"file '{frames[-1].resolve().as_posix()}'")

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def build_command(list_path: Path, out_path: Path, fps: float, codec: str,
                  crf: int, extra_filters: list[str] | None = None) -> list[str]:
    filters = list(extra_filters or [])
    # yuv420p requires even dimensions; odd output geometry otherwise fails at
    # the very last step, after all the expensive work.
    filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    cmd = [
        ffmpeg_binary(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-vf", ",".join(filters),
        "-r", str(fps),
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
           interpolate: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.parent / "frames.ffconcat"
    write_concat_list(frames, fps, list_path)

    filters = []
    if interpolate:
        filters.append(
            f"minterpolate=fps={fps * 3:g}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
        )

    cmd = build_command(list_path, out_path, fps, codec, crf, filters)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{tail}")
    return out_path
