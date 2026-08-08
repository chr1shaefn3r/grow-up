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


class TestSmoothing:
    def make(self, values) -> list[TransformParams]:
        return [TransformParams(tx=v, ty=0.0, angle=0.0, scale=1.0) for v in values]

    def test_reduces_jitter(self):
        rng = np.random.default_rng(3)
        noisy = 500.0 + rng.normal(0, 12.0, size=60)
        smoothed = align.smooth_params(self.make(noisy), window=9)

        before = np.diff(noisy).std()
        after = np.diff([p.tx for p in smoothed]).std()
        assert after < before / 2

    def test_preserves_the_underlying_trend(self):
        """Slow drift as a child grows is signal; a smoother must not flatten it."""
        trend = np.linspace(0.0, 100.0, 60)
        smoothed = align.smooth_params(self.make(trend), window=9)
        got = np.array([p.tx for p in smoothed])
        assert got[0] == pytest.approx(0.0, abs=3.0)
        assert got[-1] == pytest.approx(100.0, abs=3.0)

    def test_is_a_no_op_for_short_or_disabled_sequences(self):
        params = self.make([1.0, 2.0, 3.0])
        assert align.smooth_params(params, window=0) == params
        assert align.smooth_params(self.make([1.0, 2.0]), window=9) == self.make([1.0, 2.0])

    def test_window_larger_than_the_sequence_is_clamped(self):
        params = self.make(np.linspace(0, 10, 5))
        assert len(align.smooth_params(params, window=99)) == 5

    def test_angles_wrapping_past_pi_do_not_swing_the_mean(self):
        """Without unwrapping, a +pi/-pi crossing drags smoothed angles to zero."""
        angles = [math.pi - 0.02, math.pi - 0.01, -math.pi + 0.01, -math.pi + 0.02,
                  -math.pi + 0.03]
        params = [TransformParams(tx=0.0, ty=0.0, angle=a, scale=1.0) for a in angles]
        smoothed = align.smooth_params(params, window=5)
        for p in smoothed:
            assert abs(p.angle) > math.pi - 0.2
