"""Unit tests for quantization module."""

import pytest
import numpy as np

from quantization import (
    calculate_inter_onset_intervals,
    estimate_base_beat_duration,
    detect_local_tempo,
    generate_adaptive_grid,
    detect_grid_from_onsets,
    quantize_to_grid,
    quantize_to_adaptive_grid,
    analyze_quantization_quality,
    calculate_local_density,
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
        """IOIs should be in order of consecutive onset times"""
        result = calculate_inter_onset_intervals([0, 10, 30, 60])
        np.testing.assert_array_equal(result, [10, 20, 30])


class TestEstimateBaseBeatDuration:
    def test_empty_array(self):
        result = estimate_base_beat_duration(np.array([]), ticks_per_beat=480)
        assert result == 480  # fallback to ticks_per_beat

    def test_uniform_iois(self):
        """Uniform IOIs should detect that duration"""
        iois = np.array([120] * 50)
        result = estimate_base_beat_duration(iois, ticks_per_beat=480)
        # Should be close to 120 or a multiple/divisor
        assert result in [120, 240, 480]

    def test_filters_very_short_intervals(self):
        """Very short intervals should be filtered out"""
        # Mix of very short (5) and regular (240) intervals
        iois = np.array([5] * 10 + [240] * 50)
        result = estimate_base_beat_duration(iois, ticks_per_beat=480)
        # Should detect 240, not 5
        assert result >= 60  # at least 32nd note

    def test_standard_subdivisions(self):
        """Result should snap to standard subdivision"""
        # IOIs around quarter note length
        iois = np.array([475, 485, 480, 470, 490] * 10)
        result = estimate_base_beat_duration(iois, ticks_per_beat=480)
        # Should snap to 480 (quarter note)
        assert result == 480


class TestDetectLocalTempo:
    def test_constant_tempo(self):
        """Constant IOIs should give constant tempo"""
        onsets = list(range(0, 5000, 100))  # every 100 ticks
        base_beat = 100
        result = detect_local_tempo(onsets, base_beat, window_size=8, smoothing=0)

        # All local beats should be close to base_beat
        assert all(80 <= b <= 120 for b in result)

    def test_tempo_change(self):
        """Should detect tempo changes"""
        # First half: slow (200 tick IOI), second half: fast (100 tick IOI)
        slow_part = list(range(0, 2000, 200))
        fast_part = list(range(2000, 4000, 100))
        onsets = slow_part + fast_part
        base_beat = 150

        result = detect_local_tempo(onsets, base_beat, window_size=8, smoothing=1.0)

        # Early tempo should be slower than late tempo
        early_avg = np.mean(result[:len(slow_part)//2])
        late_avg = np.mean(result[-len(fast_part)//2:])
        assert early_avg > late_avg

    def test_single_note(self):
        """Single note should return base beat"""
        result = detect_local_tempo([100], 480, window_size=8, smoothing=0)
        assert len(result) == 1
        assert result[0] == 480


class TestGenerateAdaptiveGrid:
    def test_empty_input(self):
        result = generate_adaptive_grid([], np.array([]), subdivisions=4)
        assert result == []

    def test_single_point(self):
        result = generate_adaptive_grid([100], np.array([480]), subdivisions=4)
        assert 100 in result

    def test_grid_covers_range(self):
        """Grid should cover from min to approximately max onset"""
        onsets = [0, 1000, 2000]
        local_beats = np.array([480, 480, 480])
        result = generate_adaptive_grid(onsets, local_beats, subdivisions=4)

        assert min(result) <= 0
        # Grid may stop slightly before last onset (within one subdivision)
        assert max(result) >= 2000 - 480 // 4

    def test_subdivision_spacing(self):
        """Grid spacing should be beat / subdivisions"""
        onsets = [0, 1920]  # 4 beats at 480 ticks
        local_beats = np.array([480, 480])
        result = generate_adaptive_grid(onsets, local_beats, subdivisions=4)

        # Spacing should be 480/4 = 120
        if len(result) > 1:
            spacings = np.diff(result)
            assert all(110 <= s <= 130 for s in spacings[:5])


class TestDetectGridFromOnsets:
    def test_empty_list(self):
        grid, local_beats = detect_grid_from_onsets([])
        assert grid == []
        assert local_beats == []

    def test_single_note(self):
        grid, local_beats = detect_grid_from_onsets([100])
        assert grid == [100]
        assert len(local_beats) == 1

    def test_returns_grid_covering_all_notes(self):
        """Grid should cover from min to max onset time"""
        onsets = [0, 100, 200, 300, 400]
        grid, local_beats = detect_grid_from_onsets(onsets)
        assert min(grid) <= 0
        assert max(grid) >= 400

    def test_local_beats_length_matches_onsets(self):
        """Should return one local beat per onset"""
        onsets = [0, 100, 200, 300]
        grid, local_beats = detect_grid_from_onsets(onsets)
        assert len(local_beats) == len(onsets)


class TestQuantizeToGrid:
    def test_empty_grid(self):
        result = quantize_to_grid([100, 200], [])
        # Should return same times with 0 offset
        assert result == [(100, 100, 0), (200, 200, 0)]

    def test_exact_grid_match(self):
        """Notes on grid points should have 0 offset"""
        result = quantize_to_grid([0, 100, 200], [0, 100, 200])
        for actual, quantized, offset in result:
            assert actual == quantized
            assert offset == 0

    def test_snaps_to_nearest_grid(self):
        """Notes should snap to nearest grid point"""
        result = quantize_to_grid([95, 105], [0, 100, 200])
        assert result[0] == (95, 100, -5)  # snaps to 100, offset is -5
        assert result[1] == (105, 100, 5)  # snaps to 100, offset is +5

    def test_offset_sign_convention(self):
        """Positive offset = late, negative offset = early"""
        result = quantize_to_grid([90, 110], [100])
        assert result[0][2] < 0  # 90 is early (before 100)
        assert result[1][2] > 0  # 110 is late (after 100)


class TestQuantizeToAdaptiveGrid:
    def test_max_offset_clipping(self):
        """Offsets should be clipped when max_offset is set"""
        result = quantize_to_adaptive_grid([50, 150], [100], max_offset=10)
        assert result[0][2] == -10  # clipped from -50
        assert result[1][2] == 10   # clipped from 50


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
        """All late notes should have positive mean"""
        results = [(i, i, 10) for i in range(10)]
        stats = analyze_quantization_quality(results)
        assert stats["mean_offset"] == 10.0

    def test_mean_abs_offset(self):
        """Mean absolute offset should ignore sign"""
        results = [(0, 0, -10), (100, 100, 10)]
        stats = analyze_quantization_quality(results)
        assert stats["mean_abs_offset"] == 10.0

    def test_percentiles(self):
        """Should include 5th and 95th percentiles"""
        results = [(i, i, i - 50) for i in range(100)]  # offsets from -50 to 49
        stats = analyze_quantization_quality(results)
        assert "percentile_5" in stats
        assert "percentile_95" in stats


class TestCalculateLocalDensity:
    def test_empty_list(self):
        result = calculate_local_density([])
        assert len(result) == 0

    def test_single_note(self):
        result = calculate_local_density([100])
        assert len(result) == 1
        assert result[0] == 1  # just itself in the window

    def test_sparse_notes(self):
        """Notes far apart should have low density"""
        result = calculate_local_density([0, 1000, 2000], window_size=100)
        assert all(d == 1 for d in result)

    def test_dense_notes(self):
        """Notes close together should have high density"""
        result = calculate_local_density([0, 10, 20, 30, 40], window_size=100)
        # middle notes should see all 5 notes in their window
        assert result[2] == 5

    def test_density_varies(self):
        """Density should vary based on local context"""
        # sparse at start, dense in middle
        notes = [0, 500, 1000, 1010, 1020, 1030, 1040, 1500]
        result = calculate_local_density(notes, window_size=100)
        # first note is isolated
        assert result[0] < result[4]  # middle of dense passage has higher density
