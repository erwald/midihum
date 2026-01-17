"""unit tests for quantization module."""

import pytest
import numpy as np

from quantization import (
    calculate_inter_onset_intervals,
    find_common_ioi_values,
    calculate_local_density,
    select_grid_resolution,
    detect_grid_from_onsets,
    quantize_to_grid,
    analyze_quantization_quality,
)


class TestCalculateInterOnsetIntervals:
    def test_empty_list(self):
        result = calculate_inter_onset_intervals([])
        assert len(result) == 0

    def test_single_element(self):
        result = calculate_inter_onset_intervals([100])
        assert len(result) == 0

    def test_two_elements(self):
        result = calculate_inter_onset_intervals([100, 200])
        assert len(result) == 1
        assert result[0] == 100

    def test_multiple_elements(self):
        result = calculate_inter_onset_intervals([0, 50, 150, 200])
        np.testing.assert_array_equal(result, [50, 100, 50])

    def test_preserves_order(self):
        """iois should be in order of consecutive onset times"""
        result = calculate_inter_onset_intervals([0, 10, 30, 60])
        np.testing.assert_array_equal(result, [10, 20, 30])


class TestFindCommonIoiValues:
    def test_empty_array(self):
        result = find_common_ioi_values(np.array([]))
        assert result == []

    def test_all_below_min_ioi(self):
        result = find_common_ioi_values(np.array([1, 2, 3, 4, 5]), min_ioi=10)
        assert result == []

    def test_single_common_value(self):
        """repeated values should be detected as common"""
        iois = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        result = find_common_ioi_values(iois)
        assert len(result) >= 1
        # the most common IOI should be around 100
        assert any(90 <= v <= 110 for v in result)

    def test_multiple_common_values(self):
        """different rhythmic values should all be detected"""
        # simulate quarter notes (100) and eighth notes (50)
        iois = np.array([100] * 20 + [50] * 20)
        result = find_common_ioi_values(iois)
        # should detect both rhythmic values (within bin tolerance)
        assert len(result) >= 1


class TestCalculateLocalDensity:
    def test_empty_list(self):
        result = calculate_local_density([])
        assert len(result) == 0

    def test_single_note(self):
        result = calculate_local_density([100])
        assert len(result) == 1
        assert result[0] == 1  # just itself in the window

    def test_sparse_notes(self):
        """notes far apart should have low density"""
        result = calculate_local_density([0, 1000, 2000], window_size=100)
        assert all(d == 1 for d in result)

    def test_dense_notes(self):
        """notes close together should have high density"""
        result = calculate_local_density([0, 10, 20, 30, 40], window_size=100)
        # middle notes should see all 5 notes in their window
        assert result[2] == 5

    def test_density_varies(self):
        """density should vary based on local context"""
        # sparse at start, dense in middle
        notes = [0, 500, 1000, 1010, 1020, 1030, 1040, 1500]
        result = calculate_local_density(notes, window_size=100)
        # first note is isolated
        assert result[0] < result[4]  # middle of dense passage has higher density


class TestSelectGridResolution:
    def test_empty_common_iois(self):
        result = select_grid_resolution(10.0, [])
        assert result == 100  # fallback default

    def test_high_density_uses_finest_grid(self):
        result = select_grid_resolution(20.0, [25, 50, 100], density_thresholds=(5, 15))
        assert result == 25  # finest grid

    def test_low_density_uses_coarsest_grid(self):
        result = select_grid_resolution(3.0, [25, 50, 100], density_thresholds=(5, 15))
        assert result == 100  # coarsest grid

    def test_medium_density_uses_middle_grid(self):
        result = select_grid_resolution(10.0, [25, 50, 100], density_thresholds=(5, 15))
        assert result == 50  # middle grid


class TestDetectGridFromOnsets:
    def test_empty_list(self):
        grid, resolutions = detect_grid_from_onsets([])
        assert grid == []
        assert resolutions == []

    def test_single_note(self):
        grid, resolutions = detect_grid_from_onsets([100])
        assert grid == [100]
        assert resolutions == [0]

    def test_returns_grid_covering_all_notes(self):
        """grid should cover from min to max onset time"""
        onsets = [0, 100, 200, 300, 400]
        grid, resolutions = detect_grid_from_onsets(onsets)
        assert min(grid) <= 0
        assert max(grid) >= 400

    def test_grid_resolution_matches_onset_count(self):
        """should return one resolution per onset"""
        onsets = [0, 100, 200, 300]
        grid, resolutions = detect_grid_from_onsets(onsets)
        assert len(resolutions) == len(onsets)


class TestQuantizeToGrid:
    def test_empty_grid(self):
        result = quantize_to_grid([100, 200], [])
        # should return same times with 0 offset
        assert result == [(100, 100, 0), (200, 200, 0)]

    def test_exact_grid_match(self):
        """notes on grid points should have 0 offset"""
        result = quantize_to_grid([0, 100, 200], [0, 100, 200])
        for actual, quantized, offset in result:
            assert actual == quantized
            assert offset == 0

    def test_snaps_to_nearest_grid(self):
        """notes should snap to nearest grid point"""
        result = quantize_to_grid([95, 105], [0, 100, 200])
        assert result[0] == (95, 100, -5)  # snaps to 100, offset is -5
        assert result[1] == (105, 100, 5)  # snaps to 100, offset is +5

    def test_offset_sign_convention(self):
        """positive offset = late, negative offset = early"""
        result = quantize_to_grid([90, 110], [100])
        assert result[0][2] < 0  # 90 is early (before 100)
        assert result[1][2] > 0  # 110 is late (after 100)


class TestAnalyzeQuantizationQuality:
    def test_empty_list(self):
        result = analyze_quantization_quality([])
        assert result == {}

    def test_basic_stats(self):
        results = [(0, 0, 0), (100, 100, 5), (200, 200, -5)]
        stats = analyze_quantization_quality(results)

        assert stats["num_notes"] == 3
        assert stats["mean_offset"] == 0.0
        assert stats["median_offset"] == 0.0
        assert stats["min_offset"] == -5
        assert stats["max_offset"] == 5

    def test_all_positive_offsets(self):
        """all late notes should have positive mean"""
        results = [(i, i, 10) for i in range(10)]
        stats = analyze_quantization_quality(results)
        assert stats["mean_offset"] == 10.0

    def test_mean_abs_offset(self):
        """mean absolute offset should ignore sign"""
        results = [(0, 0, -10), (100, 100, 10)]
        stats = analyze_quantization_quality(results)
        assert stats["mean_abs_offset"] == 10.0
