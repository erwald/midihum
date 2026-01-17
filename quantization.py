"""
Cluster-based quantization for time displacement training.

This module detects chord/simultaneous-note clusters in expressive piano performances
and measures timing offsets relative to cluster centroids. This provides reliable
ground-truth for training a time displacement model.

The key insight: notes that are "meant" to be simultaneous (chords, etc.) get spread
slightly in human performance. The cluster centroid represents the "intended" beat
position, and individual note offsets represent expressive timing.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np


@dataclass
class NoteWithOffset:
    """A note with its timing offset from the cluster centroid."""
    onset_time: int
    pitch: int
    velocity: int
    offset_time: Optional[int]
    cluster_id: int
    cluster_centroid: float
    time_offset: float  # onset_time - cluster_centroid
    cluster_size: int  # number of notes in this cluster
    position_in_cluster: int  # 0 = earliest in cluster, 1 = second earliest, etc.


def cluster_onsets_by_proximity(
    onset_times: List[int],
    gap_threshold: int = 20,
) -> List[List[int]]:
    """
    Group note onsets into clusters based on temporal proximity.

    Notes within gap_threshold ticks of each other are considered
    "simultaneous" (part of the same chord or beat).

    Args:
        onset_times: sorted list of note onset times
        gap_threshold: max gap between consecutive notes in same cluster (ticks)

    Returns:
        list of clusters, where each cluster is a list of onset times
    """
    if not onset_times:
        return []

    clusters = []
    current_cluster = [onset_times[0]]

    for t in onset_times[1:]:
        if t - current_cluster[-1] <= gap_threshold:
            current_cluster.append(t)
        else:
            clusters.append(current_cluster)
            current_cluster = [t]
    clusters.append(current_cluster)

    return clusters


def compute_cluster_centroids(clusters: List[List[int]]) -> np.ndarray:
    """Compute the centroid (mean time) of each cluster."""
    return np.array([np.mean(c) for c in clusters])


def quantize_notes_to_clusters(
    notes: List[dict],
    gap_threshold: int = 20,
) -> Tuple[List[NoteWithOffset], Dict]:
    """
    Quantize notes using cluster-centroid method.

    For each note, find its cluster and compute offset from cluster centroid.
    This gives meaningful offsets for notes in multi-note clusters (chords).

    Args:
        notes: list of note dicts with onset_time, pitch, velocity, offset_time
        gap_threshold: max gap for clustering (ticks)

    Returns:
        tuple of:
        - list of NoteWithOffset objects
        - dict of statistics about the quantization
    """
    if not notes:
        return [], {}

    # Sort notes by onset time
    sorted_notes = sorted(notes, key=lambda n: n["onset_time"])
    onset_times = [n["onset_time"] for n in sorted_notes]

    # Find clusters
    clusters = cluster_onsets_by_proximity(onset_times, gap_threshold)
    centroids = compute_cluster_centroids(clusters)

    # Map each onset time to its cluster
    onset_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for t in cluster:
            onset_to_cluster[t] = cluster_id

    # Create NoteWithOffset for each note
    results = []
    for note in sorted_notes:
        t = note["onset_time"]
        cluster_id = onset_to_cluster[t]
        cluster = clusters[cluster_id]
        centroid = centroids[cluster_id]

        # Position in cluster (sorted by time)
        sorted_cluster = sorted(cluster)
        position = sorted_cluster.index(t)

        results.append(NoteWithOffset(
            onset_time=t,
            pitch=note["pitch"],
            velocity=note["velocity"],
            offset_time=note.get("offset_time"),
            cluster_id=cluster_id,
            cluster_centroid=centroid,
            time_offset=t - centroid,
            cluster_size=len(cluster),
            position_in_cluster=position,
        ))

    # Compute statistics
    all_offsets = np.array([n.time_offset for n in results])
    multi_note_offsets = np.array([n.time_offset for n in results if n.cluster_size > 1])

    stats = {
        "num_notes": len(results),
        "num_clusters": len(clusters),
        "multi_note_clusters": sum(1 for c in clusters if len(c) > 1),
        "single_note_clusters": sum(1 for c in clusters if len(c) == 1),
        "notes_in_multi_clusters": len(multi_note_offsets),
        "pct_in_multi_clusters": len(multi_note_offsets) / len(results) * 100 if results else 0,
        "all_offset_std": float(np.std(all_offsets)) if len(all_offsets) > 0 else 0,
        "all_offset_range": (int(np.min(all_offsets)), int(np.max(all_offsets))) if len(all_offsets) > 0 else (0, 0),
        "multi_offset_std": float(np.std(multi_note_offsets)) if len(multi_note_offsets) > 0 else 0,
        "multi_offset_range": (int(np.min(multi_note_offsets)), int(np.max(multi_note_offsets))) if len(multi_note_offsets) > 0 else (0, 0),
    }

    return results, stats


def get_cluster_grid(clusters: List[List[int]]) -> List[int]:
    """
    Get grid points from cluster centroids.

    Returns the centroid of each cluster as a grid point.
    """
    return [int(np.mean(c)) for c in clusters]


# Legacy API compatibility
def calculate_inter_onset_intervals(onset_times: List[int]) -> np.ndarray:
    """Calculate inter-onset intervals between consecutive note onsets."""
    if len(onset_times) < 2:
        return np.array([])
    return np.diff(np.array(onset_times))


def detect_grid_from_onsets(
    onset_times: List[int],
    ticks_per_beat: int = 480,
    subdivisions: int = 4,
    tempo_window: int = 16,
    tempo_smoothing: float = 3.0,
) -> Tuple[List[int], List[float]]:
    """
    Detect grid from note onsets using cluster-centroid method.

    Returns cluster centroids as grid points and cluster sizes as "local beats".
    """
    if len(onset_times) < 2:
        return list(onset_times), [float(ticks_per_beat)] * len(onset_times)

    clusters = cluster_onsets_by_proximity(onset_times, gap_threshold=20)
    centroids = compute_cluster_centroids(clusters)

    # Map each onset to its cluster's centroid and size
    onset_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for t in cluster:
            onset_to_cluster[t] = cluster_id

    local_values = []
    for t in onset_times:
        cluster_id = onset_to_cluster[t]
        local_values.append(float(len(clusters[cluster_id])))

    return [int(c) for c in centroids], local_values


def quantize_to_grid(
    onset_times: List[int],
    grid_times: List[int],
) -> List[Tuple[int, int, int]]:
    """
    Snap each onset to nearest grid point and calculate offset.

    Args:
        onset_times: list of actual note onset times
        grid_times: list of grid point times (cluster centroids)

    Returns:
        list of (actual_time, quantized_time, offset) tuples
    """
    if not grid_times:
        return [(t, t, 0) for t in onset_times]

    grid = np.array(grid_times)
    results = []

    for actual_time in onset_times:
        distances = np.abs(grid - actual_time)
        nearest_idx = np.argmin(distances)
        quantized_time = grid[nearest_idx]
        offset = actual_time - quantized_time
        results.append((actual_time, int(quantized_time), int(offset)))

    return results


def analyze_quantization_quality(
    quantization_results: List[Tuple[int, int, int]]
) -> dict:
    """Analyze the quality of quantization results."""
    if not quantization_results:
        return {}

    offsets = np.array([r[2] for r in quantization_results])

    return {
        "num_notes": len(offsets),
        "mean_offset": float(np.mean(offsets)),
        "std_offset": float(np.std(offsets)),
        "median_offset": float(np.median(offsets)),
        "min_offset": int(np.min(offsets)),
        "max_offset": int(np.max(offsets)),
        "mean_abs_offset": float(np.mean(np.abs(offsets))),
        "percentile_5": float(np.percentile(offsets, 5)),
        "percentile_95": float(np.percentile(offsets, 95)),
    }


# Keep these for backward compatibility
def estimate_base_beat_duration(iois: np.ndarray, ticks_per_beat: int = 480) -> int:
    """Estimate base beat duration from IOI distribution."""
    if len(iois) == 0:
        return ticks_per_beat
    return int(np.median(iois[iois >= ticks_per_beat // 8]))


def calculate_local_density(onset_times: List[int], window_size: int = 500) -> np.ndarray:
    """Calculate local note density around each onset."""
    if len(onset_times) == 0:
        return np.array([])

    times = np.array(onset_times)
    densities = np.zeros(len(times))

    for i, t in enumerate(times):
        in_window = np.sum((times >= t - window_size / 2) & (times <= t + window_size / 2))
        densities[i] = in_window

    return densities
