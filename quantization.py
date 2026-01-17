"""
adaptive grid detection for time displacement training.

this module detects the underlying rhythmic grid from expressive piano performances,
allowing us to calculate time offsets (actual_time - quantized_time) for training
the time displacement model.
"""

from typing import List, Tuple

import numpy as np
from scipy import signal


def calculate_inter_onset_intervals(onset_times: List[int]) -> np.ndarray:
    """
    calculate inter-onset intervals (IOI) between consecutive note onsets.

    args:
        onset_times: sorted list of note onset times in MIDI ticks

    returns:
        array of IOI values (length = len(onset_times) - 1)
    """
    if len(onset_times) < 2:
        return np.array([])

    times = np.array(onset_times)
    return np.diff(times)


def find_common_ioi_values(
    iois: np.ndarray, num_peaks: int = 5, min_ioi: int = 10
) -> List[int]:
    """
    find common IOI values using histogram peak detection.

    these peaks likely correspond to beat subdivisions (eighth notes, quarter notes, etc.)

    args:
        iois: array of inter-onset intervals
        num_peaks: maximum number of peaks to return
        min_ioi: minimum IOI to consider (filters out very short intervals)

    returns:
        list of common IOI values, sorted ascending
    """
    if len(iois) == 0:
        return []

    # filter out very short intervals (likely simultaneous notes or noise)
    filtered_iois = iois[iois >= min_ioi]
    if len(filtered_iois) == 0:
        return []

    # create histogram with adaptive bin size
    max_ioi = int(np.percentile(filtered_iois, 95))  # ignore outliers
    bin_width = max(10, max_ioi // 50)  # adaptive bin width
    bins = np.arange(min_ioi, max_ioi + bin_width, bin_width)

    hist, bin_edges = np.histogram(filtered_iois, bins=bins)

    # smooth histogram to reduce noise
    if len(hist) > 5:
        hist = np.convolve(hist, np.ones(3) / 3, mode="same")

    # find peaks in histogram
    peaks, properties = signal.find_peaks(hist, height=len(iois) * 0.02)

    if len(peaks) == 0:
        # fallback: use median IOI
        return [int(np.median(filtered_iois))]

    # get peak heights and sort by prominence
    peak_heights = hist[peaks]
    sorted_indices = np.argsort(peak_heights)[::-1][:num_peaks]
    top_peaks = peaks[sorted_indices]

    # convert bin indices to IOI values (use bin centers)
    common_iois = [int(bin_edges[p] + bin_width / 2) for p in sorted(top_peaks)]

    return common_iois


def calculate_local_density(
    onset_times: List[int], window_size: int = 500
) -> np.ndarray:
    """
    calculate local note density around each onset.

    density is measured as notes per unit time in a window around each note.
    higher density indicates faster passages (runs, arpeggios).

    args:
        onset_times: sorted list of note onset times
        window_size: size of window in MIDI ticks for density calculation

    returns:
        array of density values (notes per window) for each onset
    """
    if len(onset_times) == 0:
        return np.array([])

    times = np.array(onset_times)
    densities = np.zeros(len(times))

    for i, t in enumerate(times):
        # count notes within window centered on this note
        in_window = np.sum((times >= t - window_size / 2) & (times <= t + window_size / 2))
        densities[i] = in_window

    return densities


def select_grid_resolution(
    density: float, common_iois: List[int], density_thresholds: Tuple[float, float] = (5, 15)
) -> int:
    """
    select appropriate grid resolution based on local note density.

    args:
        density: local note density at this point
        common_iois: list of common IOI values (sorted ascending)
        density_thresholds: (low, high) thresholds for density classification

    returns:
        grid resolution in MIDI ticks
    """
    if not common_iois:
        return 100  # fallback default

    low_thresh, high_thresh = density_thresholds

    if density >= high_thresh:
        # high density (fast passage) -> use finest grid
        return common_iois[0]
    elif density <= low_thresh:
        # low density (slow passage) -> use coarsest grid
        return common_iois[-1]
    else:
        # medium density -> use middle grid
        mid_idx = len(common_iois) // 2
        return common_iois[mid_idx]


def detect_grid_from_onsets(
    onset_times: List[int], density_window: int = 500
) -> Tuple[List[int], List[int]]:
    """
    detect adaptive rhythmic grid from note onset times.

    this is the main entry point for grid detection. it:
    1. calculates IOIs to find common rhythmic values
    2. calculates local density to determine grid resolution per region
    3. generates grid points with adaptive resolution

    args:
        onset_times: sorted list of note onset times in MIDI ticks
        density_window: window size for density calculation

    returns:
        tuple of (grid_times, grid_resolutions):
        - grid_times: list of grid point times
        - grid_resolutions: list of grid resolution at each onset (for debugging)
    """
    if len(onset_times) < 2:
        return list(onset_times), [0] * len(onset_times)

    # step 1: find common IOI values
    iois = calculate_inter_onset_intervals(onset_times)
    common_iois = find_common_ioi_values(iois)

    if not common_iois:
        common_iois = [100]  # fallback

    # step 2: calculate local density for each note
    densities = calculate_local_density(onset_times, density_window)

    # step 3: determine grid resolution at each point and generate grid
    times = np.array(onset_times)
    min_time = times.min()
    max_time = times.max()

    # generate grid by walking through time and using local density to set resolution
    grid_times = []
    grid_resolutions = []

    # for each note, determine which grid resolution applies
    for i, t in enumerate(times):
        resolution = select_grid_resolution(densities[i], common_iois)
        grid_resolutions.append(resolution)

    # generate a unified grid that covers all notes
    # use the finest common IOI as base, but this could be refined
    base_resolution = common_iois[0]
    current_time = (min_time // base_resolution) * base_resolution

    while current_time <= max_time + base_resolution:
        grid_times.append(int(current_time))
        current_time += base_resolution

    return grid_times, grid_resolutions


def quantize_to_grid(
    onset_times: List[int], grid_times: List[int]
) -> List[Tuple[int, int, int]]:
    """
    snap each onset to nearest grid point and calculate offset.

    args:
        onset_times: list of actual note onset times
        grid_times: list of grid point times

    returns:
        list of (actual_time, quantized_time, offset) tuples
        where offset = actual_time - quantized_time
    """
    if not grid_times:
        return [(t, t, 0) for t in onset_times]

    grid = np.array(grid_times)
    results = []

    for actual_time in onset_times:
        # find nearest grid point
        distances = np.abs(grid - actual_time)
        nearest_idx = np.argmin(distances)
        quantized_time = grid[nearest_idx]
        offset = actual_time - quantized_time

        results.append((actual_time, int(quantized_time), int(offset)))

    return results


def analyze_quantization_quality(
    quantization_results: List[Tuple[int, int, int]]
) -> dict:
    """
    analyze the quality of quantization results.

    args:
        quantization_results: list of (actual_time, quantized_time, offset) tuples

    returns:
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
    }
