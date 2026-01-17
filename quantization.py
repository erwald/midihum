"""
Local beat tracking and quantization for time displacement training.

This module detects the underlying rhythmic grid from expressive piano performances
using local tempo tracking that adapts to rubato and tempo changes.
"""

from typing import List, Tuple, Optional

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def calculate_inter_onset_intervals(onset_times: List[int]) -> np.ndarray:
    """
    Calculate inter-onset intervals (IOI) between consecutive note onsets.

    Args:
        onset_times: sorted list of note onset times in MIDI ticks

    Returns:
        array of IOI values (length = len(onset_times) - 1)
    """
    if len(onset_times) < 2:
        return np.array([])

    times = np.array(onset_times)
    return np.diff(times)


def estimate_base_beat_duration(iois: np.ndarray, ticks_per_beat: int = 480) -> int:
    """
    Estimate the most likely beat duration from IOI distribution.

    Uses histogram clustering to find the dominant rhythmic unit,
    looking for common subdivisions (quarter, eighth, sixteenth notes).

    Args:
        iois: array of inter-onset intervals
        ticks_per_beat: MIDI ticks per beat (commonly 480 or 960)

    Returns:
        estimated beat duration in ticks
    """
    if len(iois) == 0:
        return ticks_per_beat

    # Filter out very short intervals (simultaneous notes, grace notes)
    # and very long intervals (pauses)
    min_ioi = ticks_per_beat // 8  # 32nd note
    max_ioi = ticks_per_beat * 4   # whole note
    filtered = iois[(iois >= min_ioi) & (iois <= max_ioi)]

    if len(filtered) == 0:
        return ticks_per_beat

    # Create histogram with fine bins
    bin_width = ticks_per_beat // 16
    bins = np.arange(min_ioi, max_ioi + bin_width, bin_width)
    hist, bin_edges = np.histogram(filtered, bins=bins)

    # Smooth histogram
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

    # Find peaks
    peaks, _ = signal.find_peaks(hist_smooth, height=np.max(hist_smooth) * 0.1)

    if len(peaks) == 0:
        return int(np.median(filtered))

    # Get the most prominent peak
    peak_heights = hist_smooth[peaks]
    best_peak = peaks[np.argmax(peak_heights)]
    estimated_ioi = int(bin_edges[best_peak] + bin_width / 2)

    # Round to nearest standard subdivision
    subdivisions = [
        ticks_per_beat // 4,   # sixteenth
        ticks_per_beat // 3,   # triplet eighth
        ticks_per_beat // 2,   # eighth
        ticks_per_beat * 2 // 3,  # triplet quarter
        ticks_per_beat,        # quarter
        ticks_per_beat * 2,    # half
    ]

    # Find closest standard subdivision
    closest = min(subdivisions, key=lambda x: abs(x - estimated_ioi))
    return closest


def detect_local_tempo(
    onset_times: List[int],
    base_beat: int,
    window_size: int = 16,
    smoothing: float = 2.0,
) -> np.ndarray:
    """
    Estimate local tempo at each note onset using windowed IOI analysis.

    This allows the tempo to vary smoothly over time, handling rubato.

    Args:
        onset_times: sorted list of note onset times
        base_beat: estimated base beat duration
        window_size: number of notes to consider for local tempo
        smoothing: gaussian smoothing sigma for tempo curve

    Returns:
        array of local beat durations at each onset
    """
    if len(onset_times) < 2:
        return np.array([base_beat] * len(onset_times))

    times = np.array(onset_times)
    iois = np.diff(times)
    n_notes = len(times)

    # For each note, estimate local tempo from surrounding IOIs
    local_beats = np.zeros(n_notes)

    for i in range(n_notes):
        # Get window of IOIs around this note
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(iois), i + window_size // 2)

        if end_idx <= start_idx:
            local_beats[i] = base_beat
            continue

        window_iois = iois[start_idx:end_idx]

        # Filter to reasonable range (0.5x to 2x base beat for subdivisions)
        min_ioi = base_beat // 4
        max_ioi = base_beat * 2
        valid_iois = window_iois[(window_iois >= min_ioi) & (window_iois <= max_ioi)]

        if len(valid_iois) == 0:
            local_beats[i] = base_beat
            continue

        # Use median IOI as local tempo estimate
        # Quantize to subdivision level
        median_ioi = np.median(valid_iois)

        # Determine which subdivision this represents
        # and extrapolate to beat level
        subdivisions = [1/4, 1/3, 1/2, 2/3, 1, 2]
        best_subdiv = min(subdivisions, key=lambda s: abs(median_ioi - base_beat * s))
        local_beats[i] = median_ioi / best_subdiv

    # Smooth the tempo curve
    if smoothing > 0 and len(local_beats) > 1:
        local_beats = gaussian_filter1d(local_beats, sigma=smoothing)

    return local_beats


def generate_adaptive_grid(
    onset_times: List[int],
    local_beats: np.ndarray,
    subdivisions: int = 4,
) -> List[int]:
    """
    Generate a grid that adapts to local tempo changes.

    Args:
        onset_times: sorted list of note onset times
        local_beats: local beat duration at each onset
        subdivisions: grid subdivisions per beat (4 = sixteenth notes)

    Returns:
        list of grid point times
    """
    if len(onset_times) == 0:
        return []

    times = np.array(onset_times)
    min_time = times.min()
    max_time = times.max()

    # Interpolate local beats to create continuous tempo function
    if len(times) > 1:
        # Create interpolation function
        from scipy.interpolate import interp1d
        tempo_func = interp1d(
            times, local_beats,
            kind='linear',
            bounds_error=False,
            fill_value=(local_beats[0], local_beats[-1])
        )
    else:
        tempo_func = lambda t: local_beats[0]

    # Generate grid by walking through time with adaptive step size
    grid_times = []
    current_time = min_time

    while current_time <= max_time:
        grid_times.append(int(current_time))

        # Get local beat duration and step by subdivision
        local_beat = tempo_func(current_time)
        step = local_beat / subdivisions
        current_time += step

    return grid_times


def quantize_to_adaptive_grid(
    onset_times: List[int],
    grid_times: List[int],
    max_offset: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Snap each onset to nearest grid point and calculate offset.

    Args:
        onset_times: list of actual note onset times
        grid_times: list of grid point times
        max_offset: if set, clip offsets to this range (helps with outliers)

    Returns:
        list of (actual_time, quantized_time, offset) tuples
        where offset = actual_time - quantized_time
    """
    if not grid_times:
        return [(t, t, 0) for t in onset_times]

    grid = np.array(grid_times)
    results = []

    for actual_time in onset_times:
        # Find nearest grid point
        distances = np.abs(grid - actual_time)
        nearest_idx = np.argmin(distances)
        quantized_time = grid[nearest_idx]
        offset = actual_time - quantized_time

        # Optionally clip offset
        if max_offset is not None:
            offset = np.clip(offset, -max_offset, max_offset)

        results.append((actual_time, int(quantized_time), int(offset)))

    return results


def detect_grid_from_onsets(
    onset_times: List[int],
    ticks_per_beat: int = 480,
    subdivisions: int = 4,
    tempo_window: int = 16,
    tempo_smoothing: float = 3.0,
) -> Tuple[List[int], List[int]]:
    """
    Detect adaptive rhythmic grid from note onset times.

    This is the main entry point for grid detection. It:
    1. Estimates base beat duration from IOI distribution
    2. Tracks local tempo variations (handles rubato)
    3. Generates an adaptive grid that follows tempo changes

    Args:
        onset_times: sorted list of note onset times in MIDI ticks
        ticks_per_beat: MIDI ticks per beat (for reference subdivisions)
        subdivisions: grid subdivisions per beat
        tempo_window: notes to consider for local tempo
        tempo_smoothing: gaussian smoothing for tempo curve

    Returns:
        tuple of (grid_times, local_beats):
        - grid_times: list of grid point times
        - local_beats: list of local beat duration at each onset
    """
    if len(onset_times) < 2:
        return list(onset_times), [ticks_per_beat] * len(onset_times)

    # Step 1: Estimate base beat duration
    iois = calculate_inter_onset_intervals(onset_times)
    base_beat = estimate_base_beat_duration(iois, ticks_per_beat)

    # Step 2: Track local tempo variations
    local_beats = detect_local_tempo(
        onset_times, base_beat,
        window_size=tempo_window,
        smoothing=tempo_smoothing,
    )

    # Step 3: Generate adaptive grid
    grid_times = generate_adaptive_grid(onset_times, local_beats, subdivisions)

    return grid_times, local_beats.tolist()


def quantize_to_grid(
    onset_times: List[int],
    grid_times: List[int],
) -> List[Tuple[int, int, int]]:
    """
    Snap each onset to nearest grid point and calculate offset.

    Convenience wrapper for quantize_to_adaptive_grid.

    Args:
        onset_times: list of actual note onset times
        grid_times: list of grid point times

    Returns:
        list of (actual_time, quantized_time, offset) tuples
    """
    return quantize_to_adaptive_grid(onset_times, grid_times)


def analyze_quantization_quality(
    quantization_results: List[Tuple[int, int, int]]
) -> dict:
    """
    Analyze the quality of quantization results.

    Args:
        quantization_results: list of (actual_time, quantized_time, offset) tuples

    Returns:
        dict with statistics about the quantization
    """
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


# Legacy exports for compatibility
def find_common_ioi_values(iois: np.ndarray, num_peaks: int = 5, min_ioi: int = 10) -> List[int]:
    """Legacy function - use estimate_base_beat_duration instead."""
    if len(iois) == 0:
        return []
    base = estimate_base_beat_duration(iois)
    return [base // 4, base // 2, base]


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
