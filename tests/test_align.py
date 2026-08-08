from __future__ import annotations

import math

import numpy as np
import pytest

from grow_up import align
from grow_up.align import TransformParams

WIDTH, HEIGHT = 1080, 1350


def apply(matrix: np.ndarray, point) -> np.ndarray:
    return matrix @ np.array([point[0], point[1], 1.0])


class TestSimilarityTransform:
    def test_lands_both_eyes_on_target(self):
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))
        left, right = np.array([412.0, 300.0]), np.array([500.0, 340.0])

        m = align.similarity_transform(left, right, dst_left, dst_right)

        assert apply(m, left) == pytest.approx(dst_left, abs=0.5)
        assert apply(m, right) == pytest.approx(dst_right, abs=0.5)

    @pytest.mark.parametrize("seed", range(25))
    def test_holds_for_arbitrary_eye_pairs(self, seed):
        """Random poses, distances and roll angles all land within half a pixel."""
        rng = np.random.default_rng(seed)
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))

        centre = rng.uniform(200, 3000, size=2)
        half = rng.uniform(20, 400)
        angle = rng.uniform(-math.pi / 4, math.pi / 4)
        offset = np.array([math.cos(angle), math.sin(angle)]) * half
        left, right = centre - offset, centre + offset

        m = align.similarity_transform(left, right, dst_left, dst_right)

        assert apply(m, left) == pytest.approx(dst_left, abs=0.5)
        assert apply(m, right) == pytest.approx(dst_right, abs=0.5)

    def test_levels_a_rolled_head(self):
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))
        left, right = np.array([400.0, 500.0]), np.array([600.0, 640.0])  # tilted

        m = align.similarity_transform(left, right, dst_left, dst_right)

        assert apply(m, left)[1] == pytest.approx(apply(m, right)[1], abs=0.5)

    def test_normalises_face_size(self):
        """Interocular normalisation is what keeps head size constant as she grows."""
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))
        target_distance = float(np.hypot(*(dst_right - dst_left)))

        for half in (30.0, 120.0, 400.0):
            left, right = np.array([500.0 - half, 400.0]), np.array([500.0 + half, 400.0])
            m = align.similarity_transform(left, right, dst_left, dst_right)
            got = float(np.hypot(*(apply(m, right) - apply(m, left))))
            assert got == pytest.approx(target_distance, abs=0.5)

    def test_coincident_eyes_are_rejected(self):
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))
        with pytest.raises(ValueError):
            align.similarity_transform(
                np.array([10.0, 10.0]), np.array([10.0, 10.0]), dst_left, dst_right
            )


class TestDecomposition:
    def test_round_trips(self):
        p = TransformParams(tx=-120.5, ty=44.25, angle=0.31, scale=2.75)
        got = align.decompose_affine(align.build_affine(p))
        assert (got.tx, got.ty, got.angle, got.scale) == pytest.approx(
            (p.tx, p.ty, p.angle, p.scale)
        )

    def test_matches_a_transform_built_from_eyes(self):
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))
        m = align.similarity_transform(
            np.array([412.0, 300.0]), np.array([500.0, 340.0]), dst_left, dst_right
        )
        assert align.build_affine(align.decompose_affine(m)) == pytest.approx(m)


class TestSequenceTransforms:
    """Regression cover for a shipped bug that put faces outside the frame.

    `stage_align` used to smooth the (tx, ty, angle, scale) series along the
    timeline. Those parameters live in each source photo's own pixel coordinate
    system, so across a real library tx ranged over thousands of pixels and the
    average belonged to no photo at all. The single-transform tests above all
    passed throughout -- only a test over a *heterogeneous batch* catches it.
    """

    def library(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            # phone close-up, DSLR wide, phone mid: different resolutions,
            # face sizes, positions and roll angles.
            (np.array([1700.0, 1300.0]), np.array([2100.0, 1330.0])),
            (np.array([3050.0, 1500.0]), np.array([3210.0, 1505.0])),
            (np.array([1500.0, 900.0]), np.array([1760.0, 915.0])),
            (np.array([240.0, 180.0]), np.array([300.0, 176.0])),
            (np.array([2600.0, 2000.0]), np.array([3100.0, 2180.0])),
        ]

    def test_every_frame_lands_its_eyes_on_target(self):
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))

        matrices = align.transforms_for(self.library(), dst_left, dst_right)

        assert len(matrices) == len(self.library())
        for (left, right), matrix in zip(self.library(), matrices):
            assert apply(matrix, left) == pytest.approx(dst_left, abs=0.5)
            assert apply(matrix, right) == pytest.approx(dst_right, abs=0.5)

    def test_every_eye_stays_inside_the_output_frame(self):
        """The symptom that surfaced: the subject was not in the picture."""
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))

        for (left, right), matrix in zip(
                self.library(), align.transforms_for(self.library(), dst_left, dst_right)):
            for eye in (apply(matrix, left), apply(matrix, right)):
                assert 0 <= eye[0] <= WIDTH, eye
                assert 0 <= eye[1] <= HEIGHT, eye

    def test_frames_are_solved_independently(self):
        """Adding a photo must not move any other photo's face."""
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))
        pairs = self.library()

        alone = align.transforms_for(pairs[:1], dst_left, dst_right)[0]
        in_company = align.transforms_for(pairs, dst_left, dst_right)[0]

        assert alone == pytest.approx(in_company)

    def test_empty_sequence(self):
        dst_left, dst_right = align.target_eyes(WIDTH, HEIGHT, (0.36, 0.42), (0.64, 0.42))
        assert align.transforms_for([], dst_left, dst_right) == []
