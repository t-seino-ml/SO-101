"""The pixel-to-arm homography, checked against a synthetic camera. No hardware."""

import numpy as np
import pytest

from so101.policy.table_frame import TableFrame

# A plausible overhead view: the camera looks down at the table from an angle, so
# the mapping is a genuine perspective transform rather than a similarity.
TRUE_MATRIX = np.array([
    [3.1e-4, 2.0e-5, 0.08],
    [1.5e-5, -3.4e-4, 0.13],
    [1.2e-5, 8.0e-5, 1.0],
])


def project(matrix, pixel):
    mapped = matrix @ np.array([pixel[0], pixel[1], 1.0])
    return mapped[:2] / mapped[2]


@pytest.fixture
def correspondences():
    rng = np.random.default_rng(0)
    pixels = rng.uniform([100, 80], [700, 520], size=(12, 2))
    positions = np.array([project(TRUE_MATRIX, pixel) for pixel in pixels])
    return pixels, positions


def test_fit_recovers_the_mapping(correspondences):
    pixels, positions = correspondences
    frame = TableFrame.fit(pixels, positions, z_table=0.02, camera="overhead")
    assert max(frame.residuals_mm) < 1.0, frame.residuals_mm


def test_round_trip_through_pixel_space(correspondences):
    pixels, positions = correspondences
    frame = TableFrame.fit(pixels, positions)
    for pixel in pixels:
        back = frame.to_pixel(frame.to_arm(pixel))
        assert np.allclose(back, pixel, atol=1e-3), f"{back} vs {pixel}"


def test_one_bad_correspondence_is_rejected(correspondences):
    """A mis-touched block must not drag the whole calibration."""
    pixels, positions = correspondences
    positions = positions.copy()
    positions[3] += [0.05, -0.04]      # 60 mm off, a plausible human error

    frame = TableFrame.fit(pixels, positions)
    good = [r for i, r in enumerate(frame.residuals_mm) if i != 3]
    assert max(good) < 2.0, f"outlier contaminated the fit: {frame.residuals_mm}"


def test_too_few_points_is_refused():
    with pytest.raises(ValueError, match="at least"):
        TableFrame.fit([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]])


def test_reach_target_uses_the_table_height(correspondences):
    pixels, positions = correspondences
    frame = TableFrame.fit(pixels, positions, z_table=0.025)
    target = frame.reach_target(pixels[0])
    assert target[2] == pytest.approx(0.025)
    assert np.allclose(target[:2], positions[0], atol=1e-3)


def test_save_and_load_round_trip(correspondences, tmp_path):
    pixels, positions = correspondences
    frame = TableFrame.fit(pixels, positions, z_table=0.02, camera="overhead")
    path = frame.save(tmp_path / "table_frame.json")

    loaded = TableFrame.load(path)
    assert loaded.camera == "overhead"
    assert loaded.z_table == pytest.approx(0.02)
    assert np.allclose(loaded.to_arm(pixels[0]), frame.to_arm(pixels[0]))
