"""Unit tests for cluster-based quantization module."""

import pytest
import numpy as np

from quantization import (
    cluster_onsets_by_proximity,
    compute_cluster_centroids,
    quantize_notes_to_clusters,
    calculate_inter_onset_intervals,
    quantize_to_grid,
    analyze_quantization_quality,
    calculate_local_density,
    NoteWithOffset,
)


class TestClusterOnsetsByProximity:
    def test_empty_list(self):
        result = cluster_onsets_by_proximity([])
        assert result == []

    def test_single_element(self):
        result = cluster_onsets_by_proximity([100])
        assert result == [[100]]

    def test_two_close_elements(self):
        """Elements within threshold should be in same cluster."""
        result = cluster_onsets_by_proximity([100, 110], gap_threshold=20)
        assert result == [[100, 110]]

    def test_two_far_elements(self):
        """Elements beyond threshold should be in different clusters."""
        result = cluster_onsets_by_proximity([100, 200], gap_threshold=20)
        assert result == [[100], [200]]

    def test_multiple_clusters(self):
        """Should create correct clusters."""
        times = [0, 5, 10, 100, 105, 200]
        result = cluster_onsets_by_proximity(times, gap_threshold=20)
        assert len(result) == 3
        assert result[0] == [0, 5, 10]
        assert result[1] == [100, 105]
        assert result[2] == [200]

    def test_chained_clustering(self):
        """Notes that chain together should form one cluster."""
        # Each note is within 15 of the previous, threshold is 20
        times = [0, 15, 30, 45, 60]
        result = cluster_onsets_by_proximity(times, gap_threshold=20)
        assert len(result) == 1
        assert result[0] == times


class TestComputeClusterCentroids:
    def test_single_note_clusters(self):
        clusters = [[100], [200], [300]]
        result = compute_cluster_centroids(clusters)
        np.testing.assert_array_equal(result, [100, 200, 300])

    def test_multi_note_clusters(self):
        clusters = [[0, 10, 20], [100, 110]]
        result = compute_cluster_centroids(clusters)
        np.testing.assert_array_almost_equal(result, [10, 105])

    def test_mixed_clusters(self):
        clusters = [[50], [100, 110, 120]]
        result = compute_cluster_centroids(clusters)
        np.testing.assert_array_almost_equal(result, [50, 110])


class TestQuantizeNotesToClusters:
    def test_empty_notes(self):
        result, stats = quantize_notes_to_clusters([])
        assert result == []
        assert stats == {}

    def test_single_note(self):
        notes = [{"onset_time": 100, "pitch": 60, "velocity": 80}]
        result, stats = quantize_notes_to_clusters(notes)

        assert len(result) == 1
        assert result[0].onset_time == 100
        assert result[0].time_offset == 0  # single note is its own centroid
        assert result[0].cluster_size == 1
        assert stats["num_clusters"] == 1
        assert stats["single_note_clusters"] == 1

    def test_chord_spread(self):
        """Notes in a chord should have offsets from centroid."""
        notes = [
            {"onset_time": 100, "pitch": 60, "velocity": 80},
            {"onset_time": 105, "pitch": 64, "velocity": 80},
            {"onset_time": 110, "pitch": 67, "velocity": 80},
        ]
        result, stats = quantize_notes_to_clusters(notes, gap_threshold=20)

        assert len(result) == 3
        assert stats["num_clusters"] == 1
        assert stats["multi_note_clusters"] == 1
        assert stats["notes_in_multi_clusters"] == 3

        # Centroid should be 105
        centroid = 105
        offsets = [n.time_offset for n in result]
        assert offsets[0] == 100 - centroid  # -5
        assert offsets[1] == 105 - centroid  # 0
        assert offsets[2] == 110 - centroid  # 5

    def test_position_in_cluster(self):
        """Position should be ordered by time."""
        notes = [
            {"onset_time": 110, "pitch": 67, "velocity": 80},
            {"onset_time": 100, "pitch": 60, "velocity": 80},
            {"onset_time": 105, "pitch": 64, "velocity": 80},
        ]
        result, stats = quantize_notes_to_clusters(notes, gap_threshold=20)

        # Sort by onset time to check positions
        sorted_result = sorted(result, key=lambda n: n.onset_time)
        assert sorted_result[0].position_in_cluster == 0
        assert sorted_result[1].position_in_cluster == 1
        assert sorted_result[2].position_in_cluster == 2


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


class TestQuantizeToGrid:
    def test_empty_grid(self):
        result = quantize_to_grid([100, 200], [])
        assert result == [(100, 100, 0), (200, 200, 0)]

    def test_exact_grid_match(self):
        result = quantize_to_grid([0, 100, 200], [0, 100, 200])
        for actual, quantized, offset in result:
            assert actual == quantized
            assert offset == 0

    def test_snaps_to_nearest_grid(self):
        result = quantize_to_grid([95, 105], [0, 100, 200])
        assert result[0] == (95, 100, -5)
        assert result[1] == (105, 100, 5)

    def test_offset_sign_convention(self):
        """Positive offset = late, negative offset = early."""
        result = quantize_to_grid([90, 110], [100])
        assert result[0][2] < 0  # 90 is early
        assert result[1][2] > 0  # 110 is late


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

    def test_mean_abs_offset(self):
        results = [(0, 0, -10), (100, 100, 10)]
        stats = analyze_quantization_quality(results)
        assert stats["mean_abs_offset"] == 10.0


class TestCalculateLocalDensity:
    def test_empty_list(self):
        result = calculate_local_density([])
        assert len(result) == 0

    def test_single_note(self):
        result = calculate_local_density([100])
        assert len(result) == 1
        assert result[0] == 1

    def test_sparse_notes(self):
        result = calculate_local_density([0, 1000, 2000], window_size=100)
        assert all(d == 1 for d in result)

    def test_dense_notes(self):
        result = calculate_local_density([0, 10, 20, 30, 40], window_size=100)
        assert result[2] == 5
