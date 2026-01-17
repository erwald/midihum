"""
test script for quantization and visualization.

run with: python test_quantization.py
"""

from pathlib import Path
import numpy as np

from quantization import (
    calculate_inter_onset_intervals,
    find_common_ioi_values,
    calculate_local_density,
    detect_grid_from_onsets,
    quantize_to_grid,
    analyze_quantization_quality,
)
from plotter import plot_piano_roll_with_grid, plot_quantization_analysis


def generate_synthetic_performance(
    base_tempo: int = 100,
    num_measures: int = 8,
    beats_per_measure: int = 4,
    humanize_std: int = 15,
):
    """
    generate synthetic note onset times simulating an expressive performance.

    creates a mix of:
    - quarter notes (on beats)
    - eighth note runs
    - chords
    with humanized timing (gaussian noise added to perfect timing).

    returns:
        list of note dicts with onset_time, offset_time, pitch, velocity
    """
    notes = []
    ticks_per_beat = base_tempo

    # generate notes for each measure
    current_time = 0
    note_id = 0

    for measure in range(num_measures):
        measure_start = measure * beats_per_measure * ticks_per_beat

        if measure % 4 == 0:
            # quarter notes on each beat (melody)
            for beat in range(beats_per_measure):
                perfect_time = measure_start + beat * ticks_per_beat
                humanized_time = int(perfect_time + np.random.normal(0, humanize_std))
                pitch = 60 + np.random.choice([0, 2, 4, 5, 7, 9, 11, 12])  # C major scale

                notes.append({
                    "onset_time": humanized_time,
                    "offset_time": humanized_time + int(ticks_per_beat * 0.9),
                    "pitch": pitch,
                    "velocity": np.random.randint(60, 100),
                })
                note_id += 1

        elif measure % 4 == 1:
            # eighth note run (fast passage)
            for eighth in range(beats_per_measure * 2):
                perfect_time = measure_start + eighth * (ticks_per_beat // 2)
                humanized_time = int(perfect_time + np.random.normal(0, humanize_std * 0.7))
                pitch = 60 + eighth  # ascending run

                notes.append({
                    "onset_time": humanized_time,
                    "offset_time": humanized_time + int(ticks_per_beat * 0.4),
                    "pitch": pitch,
                    "velocity": np.random.randint(50, 80),
                })
                note_id += 1

        elif measure % 4 == 2:
            # chords on beats 1 and 3
            for beat in [0, 2]:
                perfect_time = measure_start + beat * ticks_per_beat
                humanized_time = int(perfect_time + np.random.normal(0, humanize_std))

                # chord (3-4 notes at once, slightly spread)
                for i, pitch in enumerate([48, 52, 55, 60]):
                    spread = int(np.random.normal(0, 5))  # slight spread for chord
                    notes.append({
                        "onset_time": humanized_time + spread,
                        "offset_time": humanized_time + spread + int(ticks_per_beat * 1.8),
                        "pitch": pitch,
                        "velocity": np.random.randint(70, 100),
                    })
                    note_id += 1

        else:
            # sixteenth note run (very fast passage)
            for sixteenth in range(beats_per_measure * 4):
                perfect_time = measure_start + sixteenth * (ticks_per_beat // 4)
                humanized_time = int(perfect_time + np.random.normal(0, humanize_std * 0.5))
                pitch = 72 - (sixteenth % 8)  # descending pattern

                notes.append({
                    "onset_time": humanized_time,
                    "offset_time": humanized_time + int(ticks_per_beat * 0.2),
                    "pitch": pitch,
                    "velocity": np.random.randint(40, 70),
                })
                note_id += 1

    # sort by onset time
    notes.sort(key=lambda n: n["onset_time"])

    return notes


def test_quantization_with_synthetic_data():
    """run quantization tests with synthetic data."""
    print("generating synthetic performance data...")
    np.random.seed(42)  # for reproducibility

    notes = generate_synthetic_performance(
        base_tempo=100,
        num_measures=8,
        humanize_std=12,
    )

    print(f"generated {len(notes)} notes")

    # extract onset times
    onset_times = [n["onset_time"] for n in notes]

    # test IOI calculation
    print("\ncalculating inter-onset intervals...")
    iois = calculate_inter_onset_intervals(onset_times)
    print(f"  IOI stats: min={iois.min():.0f}, max={iois.max():.0f}, median={np.median(iois):.0f}")

    # test common IOI detection
    print("\nfinding common IOI values...")
    common_iois = find_common_ioi_values(iois)
    print(f"  common IOIs: {common_iois}")

    # test density calculation
    print("\ncalculating local density...")
    densities = calculate_local_density(onset_times)
    print(f"  density stats: min={densities.min():.1f}, max={densities.max():.1f}, mean={densities.mean():.1f}")

    # test grid detection
    print("\ndetecting grid from onsets...")
    grid_times, grid_resolutions = detect_grid_from_onsets(onset_times)
    print(f"  detected {len(grid_times)} grid points")
    print(f"  grid resolution range: {min(grid_resolutions)} to {max(grid_resolutions)}")

    # test quantization
    print("\nquantizing to grid...")
    quant_results = quantize_to_grid(onset_times, grid_times)

    # analyze quality
    print("\nanalyzing quantization quality...")
    quality = analyze_quantization_quality(quant_results)
    for key, value in quality.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # generate visualizations
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    print(f"\ngenerating visualizations in {output_dir}/...")
    plot_quantization_analysis(
        notes,
        grid_times,
        quant_results,
        output_dir,
        name="synthetic_test",
    )

    # also generate a zoomed view of first 2 measures
    print("generating zoomed view...")
    notes_with_offsets = []
    for note, qr in zip(notes, quant_results):
        note_copy = note.copy()
        note_copy["time_offset"] = qr[2]
        notes_with_offsets.append(note_copy)

    plot_piano_roll_with_grid(
        notes_with_offsets,
        grid_times,
        output_dir / "synthetic_test_zoomed.png",
        time_range=(0, 800),
        title="Piano Roll - First 2 Measures (Zoomed)",
    )

    print("\ndone! check test_output/ for visualizations.")


if __name__ == "__main__":
    test_quantization_with_synthetic_data()
