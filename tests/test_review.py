from __future__ import annotations

import json
import re

import pytest

from grow_up import db, review


@pytest.fixture()
def conn(tmp_path):
    conn = db.connect(tmp_path / "test.sqlite")
    stamp = "2026-01-01T00:00:00.000Z"
    for i in range(1, 4):
        asset_id = f"asset-{i}"
        conn.execute("INSERT INTO assets (id, local_datetime, indexed_at) VALUES (?, ?, ?)",
                     (asset_id, f"2026-0{i}-10T10:00:00.000Z", stamp))
        conn.execute("INSERT INTO selection (asset_id, bucket, rank, selected_at)"
                     " VALUES (?, ?, 0, ?)", (asset_id, f"2026-0{i}", stamp))
        conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at) VALUES (?, ?, ?, ?)",
                     (asset_id, str(tmp_path / "frames" / f"frame_{i:06d}.jpg"), i, stamp))
    return conn


def add_rejected(conn, asset_id: str, reason: str, tmp_path) -> None:
    stamp = "2026-01-01T00:00:00.000Z"
    conn.execute("INSERT INTO assets (id, local_datetime, indexed_at) VALUES (?, ?, ?)",
                 (asset_id, "2026-04-10T10:00:00.000Z", stamp))
    conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                 " VALUES (?, ?, 'original', ?)",
                 (asset_id, str(tmp_path / "originals" / f"{asset_id}.jpg"), stamp))
    conn.execute("INSERT INTO metrics (asset_id, detected, reject_reason, analyzed_at)"
                 " VALUES (?, 1, ?, ?)", (asset_id, reason, stamp))


class TestContactSheet:
    def test_lists_every_selected_frame_in_order(self, conn, tmp_path):
        out = tmp_path / "out" / "contact-sheet.html"
        assert review.write_contact_sheet(conn, out) == 3

        html = out.read_text()
        assert html.count("<figure") == 3
        assert html.index("asset-1") < html.index("asset-2") < html.index("asset-3")

    def test_is_self_contained(self, conn, tmp_path):
        """The page is opened over file://, so any remote asset simply fails to load."""
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        html = out.read_text()

        assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
        assert "<script" in html and "cdn" not in html.lower()

    def test_references_frames_by_relative_path(self, conn, tmp_path):
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        html = out.read_text()

        assert "../frames/frame_000001.jpg" in html
        assert str(tmp_path) not in html, "absolute paths break if the folder moves"

    def test_carries_the_asset_id_for_the_rejects_file(self, conn, tmp_path):
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        assert 'data-id="asset-2"' in out.read_text()

    def test_tiles_are_big_enough_to_judge_a_face(self, conn, tmp_path):
        """A 1080x1350 frame rendered ~150px wide shows nothing useful about
        eye alignment or sharpness, which is the whole point of the page."""
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)

        default_tile = re.search(r"--tile:\s*(\d+)px", out.read_text())
        assert default_tile and int(default_tile.group(1)) >= 300

    def test_images_are_shown_whole_never_cropped(self, conn, tmp_path):
        """object-fit: cover would crop away the framing being assessed."""
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        css = out.read_text()

        assert re.search(r"figure img\s*\{[^}]*width:100%[^}]*height:auto", css)
        assert "object-fit:cover" not in css.replace(" ", "")

    def test_offers_size_controls(self, conn, tmp_path):
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        html = out.read_text()

        assert html.count("data-tile=") == 3
        assert 'data-tile="520px"' in html

    def test_has_a_full_size_viewer(self, conn, tmp_path):
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        html = out.read_text()

        assert 'id="viewer"' in html and 'id="viewer-img"' in html
        assert "ArrowRight" in html and "ArrowLeft" in html, "flipbook navigation"
        assert "Escape" in html

    def test_rejection_has_its_own_control(self, conn, tmp_path):
        """Clicking the image opens the viewer, so rejecting needs a separate
        affordance rather than sharing the click."""
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        html = out.read_text()

        assert html.count(">reject<") == 3
        assert "stopPropagation" in html

    def test_each_frame_carries_a_label_for_the_viewer(self, conn, tmp_path):
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        assert 'data-label="2026-01-10  #1"' in out.read_text()


class TestRejectsGallery:
    def test_groups_by_reason(self, conn, tmp_path):
        add_rejected(conn, "bad-1", "head_turned", tmp_path)
        add_rejected(conn, "bad-2", "head_turned", tmp_path)
        add_rejected(conn, "bad-3", "blurry", tmp_path)

        out = tmp_path / "out" / "rejects.html"
        assert review.write_rejects_gallery(conn, out) == 3

        html = out.read_text()
        assert "head_turned" in html and "blurry" in html
        assert "(2)" in html

    def test_samples_rather_than_rendering_everything(self, conn, tmp_path):
        """A sample per reason answers 'are my thresholds sane'; 4000 originals do not."""
        for i in range(30):
            add_rejected(conn, f"bad-{i}", "blurry", tmp_path)

        out = tmp_path / "out" / "rejects.html"
        total = review.write_rejects_gallery(conn, out, limit_per_reason=5)

        assert total == 5
        assert "showing 5 of 30" in out.read_text()

    def test_survives_assets_that_were_never_downloaded(self, conn, tmp_path):
        stamp = "2026-01-01T00:00:00.000Z"
        conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                     " VALUES ('nofile', '2026-04-10T10:00:00.000Z', ?)", (stamp,))
        conn.execute("INSERT INTO metrics (asset_id, detected, reject_reason, analyzed_at)"
                     " VALUES ('nofile', 0, 'no_face_detected', ?)", (stamp,))

        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out)
        assert "no_face_detected" in out.read_text()

    def test_empty_database_still_writes_a_page(self, conn, tmp_path):
        out = tmp_path / "out" / "rejects.html"
        assert review.write_rejects_gallery(conn, out) == 0
        assert out.exists()

    def test_also_gets_the_viewer_and_size_controls(self, conn, tmp_path):
        """Rejects are full-resolution originals, so judging them at tile size
        is even less workable than judging the aligned crops."""
        add_rejected(conn, "bad-1", "blurry", tmp_path)
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out)
        html = out.read_text()

        assert 'id="viewer"' in html
        assert "data-tile=" in html
        assert 'data-id="bad-1"' in html

    def test_stays_self_contained(self, conn, tmp_path):
        add_rejected(conn, "bad-1", "blurry", tmp_path)
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out)
        html = out.read_text()

        assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)


class TestManualRejects:
    def test_missing_file_means_no_rejects(self, tmp_path):
        assert review.load_manual_rejects(tmp_path / "rejects.json") == set()

    def test_reads_the_shape_the_contact_sheet_writes(self, tmp_path):
        path = tmp_path / "rejects.json"
        path.write_text(json.dumps({"rejected": ["a", "b"]}))
        assert review.load_manual_rejects(path) == {"a", "b"}

    def test_accepts_a_bare_list(self, tmp_path):
        path = tmp_path / "rejects.json"
        path.write_text(json.dumps(["a", "b"]))
        assert review.load_manual_rejects(path) == {"a", "b"}
