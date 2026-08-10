from __future__ import annotations

import pytest

from grow_up.immich import AspectMismatch, Face, pick_face, scale_bbox

SUBJECT = "11111111-1111-4111-8111-111111111111"
SIBLING = "22222222-2222-4222-8222-222222222222"


def face_json(person_id: str | None, box=(100, 120, 200, 240), size=(1440, 1080)):
    return {
        "id": "face-1",
        "boundingBoxX1": box[0], "boundingBoxY1": box[1],
        "boundingBoxX2": box[2], "boundingBoxY2": box[3],
        "imageWidth": size[0], "imageHeight": size[1],
        "sourceType": "machine-learning",
        "person": None if person_id is None else {"id": person_id, "name": "x"},
    }


class TestPickFace:
    def test_finds_her_among_several_people(self):
        """Immich's own person association is why no face recognition is needed here."""
        faces = [face_json(SIBLING), face_json(SUBJECT, box=(300, 100, 400, 220)), face_json(None)]
        face, n = pick_face(faces, SUBJECT)
        assert face is not None and (face.x1, face.y1) == (300, 100)
        assert n == 1

    def test_returns_none_when_she_is_not_detected(self):
        face, n = pick_face([face_json(SIBLING), face_json(None)], SUBJECT)
        assert face is None and n == 0

    def test_empty_face_list(self):
        assert pick_face([], SUBJECT) == (None, 0)

    def test_largest_wins_when_a_merge_left_duplicates(self):
        faces = [
            face_json(SUBJECT, box=(0, 0, 50, 50)),
            face_json(SUBJECT, box=(100, 100, 400, 400)),
        ]
        face, n = pick_face(faces, SUBJECT)
        assert face is not None and face.area == 300 * 300
        assert n == 2, "the count is returned so the caller can flag the duplicate"

    def test_tolerates_a_null_person(self):
        assert pick_face([face_json(None)], SUBJECT) == (None, 0)


class TestScaleBbox:
    def test_maps_detection_space_onto_the_original(self):
        """Immich reports boxes against a preview, not the full-resolution file."""
        face = Face(100, 120, 200, 240, image_width=1440, image_height=1080, source_type=None)
        assert scale_bbox(face, 4032, 3024) == (280, 336, 560, 672)

    def test_identity_when_sizes_match(self):
        face = Face(10, 20, 30, 40, image_width=800, image_height=600, source_type=None)
        assert scale_bbox(face, 800, 600) == (10, 20, 30, 40)

    def test_rejects_a_rotated_image(self):
        """A portrait original against a landscape detection space means the decode
        skipped EXIF orientation; cropping anyway would land on the wrong face."""
        face = Face(100, 120, 200, 240, image_width=1440, image_height=1080, source_type=None)
        with pytest.raises(AspectMismatch):
            scale_bbox(face, 3024, 4032)

    def test_tolerates_rounding_differences_in_aspect(self):
        # Preview generation rounds dimensions, so exact equality is too strict.
        face = Face(10, 10, 100, 100, image_width=1439, image_height=1080, source_type=None)
        assert scale_bbox(face, 4032, 3024)

    def test_rejects_missing_dimensions(self):
        face = Face(10, 10, 100, 100, image_width=0, image_height=0, source_type=None)
        with pytest.raises(AspectMismatch):
            scale_bbox(face, 4032, 3024)
