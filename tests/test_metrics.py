from __future__ import annotations

import math

import numpy as np
import pytest

from grow_up import metrics
from grow_up.metrics import FaceMetrics

LIMITS = {
    "max_yaw": 20.0, "max_pitch": 18.0, "max_roll": 25.0, "max_gaze": 0.35,
    "max_blink": 0.45, "max_oob_frac": 0.005, "allow_bbox_clipped": False,
    "min_interocular_px": 60.0, "min_sharpness": 12.0,
    "min_exposure_lo": 8.0, "max_exposure_hi": 250.0,
}
WEIGHTS = {"w_pose": 1.0, "w_gaze": 1.0, "w_eyes_open": 1.0,
           "w_sharpness": 1.0, "w_size": 0.5}


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return rz @ ry @ rx


class TestEuler:
    @pytest.mark.parametrize("angles", [(0, 0, 0), (15, -10, 5), (-30, 20, -12), (5, 5, 5)])
    def test_round_trips(self, angles):
        got = metrics.euler_from_matrix(rotation(*angles))
        assert got == pytest.approx(angles, abs=1e-6)

    def test_ignores_scale_baked_into_the_matrix(self):
        """MediaPipe's transformation matrix carries scale as well as rotation."""
        angles = (12.0, -7.0, 3.0)
        assert metrics.euler_from_matrix(rotation(*angles) * 3.7) == pytest.approx(angles, abs=1e-6)

    def test_accepts_the_4x4_mediapipe_emits(self):
        m = np.eye(4)
        m[:3, :3] = rotation(10, 4, -2)
        m[:3, 3] = [12.0, -3.0, 40.0]
        assert metrics.euler_from_matrix(m) == pytest.approx((10, 4, -2), abs=1e-6)

    def test_rejects_a_degenerate_matrix(self):
        with pytest.raises(ValueError):
            metrics.euler_from_matrix(np.zeros((3, 3)))


class TestOutOfFrame:
    def landmarks(self, pts):
        return np.array(pts, dtype=np.float64)

    def test_fully_inside_is_zero(self):
        pts = self.landmarks([[10, 10], [50, 50], [90, 90]])
        assert metrics.out_of_bounds_fraction(pts, 100, 100) == 0.0

    def test_extrapolated_mesh_is_caught(self):
        """A half-out-of-frame face still yields 478 points; their positions give it away."""
        pts = self.landmarks([[10, 10], [50, 50], [-5, 40], [120, 40]])
        assert metrics.out_of_bounds_fraction(pts, 100, 100) == pytest.approx(0.5)

    def test_inset_tightens_the_test(self):
        pts = self.landmarks([[3, 50], [50, 50]])
        assert metrics.out_of_bounds_fraction(pts, 100, 100) == 0.0
        assert metrics.out_of_bounds_fraction(pts, 100, 100, inset=5) == pytest.approx(0.5)


def test_bbox_clipping():
    assert metrics.bbox_is_clipped((0, 40, 60, 90), 200, 200)
    assert metrics.bbox_is_clipped((40, 40, 199, 90), 200, 200)
    assert not metrics.bbox_is_clipped((40, 40, 160, 160), 200, 200)


class TestSharpness:
    def test_blur_scores_below_detail(self):
        rng = np.random.default_rng(0)
        detailed = rng.integers(0, 255, size=(64, 64)).astype(np.float64)
        blurred = np.repeat(np.repeat(detailed[::8, ::8], 8, axis=0), 8, axis=1)
        assert metrics.laplacian_variance(detailed) > metrics.laplacian_variance(blurred)

    def test_flat_patch_is_zero(self):
        assert metrics.laplacian_variance(np.full((32, 32), 128.0)) == pytest.approx(0.0)

    def test_tiny_patch_does_not_raise(self):
        assert metrics.laplacian_variance(np.zeros((2, 2))) == 0.0


class TestIrisCenters:
    def test_orders_left_to_right_in_image_space(self):
        pts = np.zeros((metrics.N_LANDMARKS, 3))
        pts[metrics.IRIS_CENTER_A] = [700, 300, 0]
        pts[metrics.IRIS_CENTER_B] = [400, 310, 0]
        left, right = metrics.iris_centers(pts)
        assert left[0] < right[0]
        assert left[0] == 400 and right[0] == 700


class TestHardReject:
    def passing(self, **overrides) -> FaceMetrics:
        base = dict(detected=1, yaw=2.0, pitch=1.0, roll=3.0, gaze_x=0.05, gaze_y=0.02,
                    blink_l=0.05, blink_r=0.04, oob_frac=0.0, bbox_clipped=0,
                    interocular_px=140.0, sharpness=90.0, exposure_lo=40.0, exposure_hi=210.0)
        base.update(overrides)
        return FaceMetrics(**base)

    def test_a_good_frame_passes(self):
        assert metrics.hard_reject(self.passing(), LIMITS) is None

    @pytest.mark.parametrize("overrides,reason", [
        ({"detected": 0}, "no_face_detected"),
        ({"bbox_clipped": 1}, "bbox_clipped"),
        ({"oob_frac": 0.2}, "partially_out_of_frame"),
        ({"yaw": 45.0}, "head_turned"),
        ({"pitch": -40.0}, "head_tilted"),
        ({"roll": 50.0}, "head_rolled"),
        ({"gaze_x": 0.9}, "looking_away"),
        ({"gaze_y": -0.9}, "looking_away"),
        ({"blink_l": 0.95}, "eyes_closed"),
        ({"interocular_px": 20.0}, "face_too_small"),
        ({"sharpness": 1.0}, "blurry"),
        ({"exposure_hi": 254.0}, "overexposed"),
        ({"exposure_lo": 1.0}, "underexposed"),
    ])
    def test_each_filter_reports_its_own_reason(self, overrides, reason):
        assert metrics.hard_reject(self.passing(**overrides), LIMITS) == reason

    def test_head_on_but_eyes_averted_is_rejected(self):
        """The case head pose alone cannot catch, and the whole point of blendshapes."""
        frame = self.passing(yaw=0.0, pitch=0.0, roll=0.0, gaze_x=0.8)
        assert metrics.hard_reject(frame, LIMITS) == "looking_away"

    def test_clipping_can_be_allowed(self):
        limits = {**LIMITS, "allow_bbox_clipped": True}
        assert metrics.hard_reject(self.passing(bbox_clipped=1), limits) is None


class TestScore:
    def test_stays_within_unit_range(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            m = FaceMetrics(
                detected=1,
                yaw=rng.uniform(-20, 20), pitch=rng.uniform(-18, 18), roll=rng.uniform(-25, 25),
                gaze_x=rng.uniform(-0.35, 0.35), gaze_y=rng.uniform(-0.35, 0.35),
                blink_l=rng.uniform(0, 0.45), blink_r=rng.uniform(0, 0.45),
                interocular_px=rng.uniform(60, 400), sharpness=rng.uniform(12, 500),
            )
            assert 0.0 <= metrics.composite_score(m, LIMITS, WEIGHTS) <= 1.0

    def test_a_head_on_sharp_frame_outranks_a_turned_blurry_one(self):
        good = FaceMetrics(detected=1, yaw=1.0, pitch=1.0, roll=1.0, gaze_x=0.0, gaze_y=0.0,
                           blink_l=0.0, blink_r=0.0, interocular_px=300.0, sharpness=400.0)
        poor = FaceMetrics(detected=1, yaw=19.0, pitch=17.0, roll=24.0, gaze_x=0.34, gaze_y=0.3,
                           blink_l=0.4, blink_r=0.4, interocular_px=62.0, sharpness=13.0)
        assert (metrics.composite_score(good, LIMITS, WEIGHTS)
                > metrics.composite_score(poor, LIMITS, WEIGHTS))

    def test_zero_weights_do_not_divide_by_zero(self):
        m = FaceMetrics(detected=1, yaw=0.0, pitch=0.0, roll=0.0)
        assert metrics.composite_score(m, LIMITS, dict.fromkeys(WEIGHTS, 0.0)) == 0.0
