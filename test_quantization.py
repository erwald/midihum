"""
Test script for quantization and visualization using real MIDI data.

Run with: python test_quantization.py
"""

import random
from pathlib import Path

import numpy as np
from mido import MidiFile

from quantization import (
    calculate_inter_onset_intervals,
    find_common_ioi_values,
    calculate_local_density,
    detect_grid_from_onsets,
    quantize_to_grid,
    analyze_quantization_quality,
)
from plotter import plot_piano_roll_with_grid, plot_quantization_analysis
from midi_utility import get_note_tracks, get_midi_filepaths


def load_notes_from_midi(midi_path: Path) -> list:
    """
    Load notes from a MIDI file and return them in the format expected by plotting.

    Returns:
        list of note dicts with onset_time, offset_time, pitch, velocity
    """
    midi_file = MidiFile(midi_path)
    tracks = get_note_tracks(midi_file)

    notes = []
    # Track active notes to pair note_on with note_off
    active_notes = {}  # (track_idx, pitch) -> note_dict

    for track in tracks:
        for event in track.note_events:
            key = (track.index, event.note)

            if event.type == "note_on" and event.velocity > 0:
                # Start of a note
                active_notes[key] = {
                    "onset_time": event.time,
                    "pitch": event.note,
                    "velocity": event.velocity,
                }
            elif event.type == "note_off" or (event.type == "note_on" and event.velocity == 0):
                # End of a note
                if key in active_notes:
                    note = active_notes.pop(key)
                    note["offset_time"] = event.time
                    notes.append(note)

    # Handle any notes that didn't get a note_off (add default duration)
    for note in active_notes.values():
        note["offset_time"] = note["onset_time"] + 100
        notes.append(note)

    # Sort by onset time
    notes.sort(key=lambda n: n["onset_time"])

    return notes


def test_quantization_with_real_midi():
    """Run quantization tests with real MIDI data from the training set."""
    midi_dir = Path("midi_data_repaired_cache")

    if not midi_dir.exists():
        print(f"error: {midi_dir} not found")
        return

    midi_files = get_midi_filepaths(midi_dir)
    if not midi_files:
        print(f"error: no MIDI files found in {midi_dir}")
        return

    # Use a fixed seed for reproducibility when selecting files
    random.seed(42)
    selected_files = random.sample(midi_files, min(3, len(midi_files)))

    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    for midi_path in selected_files:
        print(f"\n{'='*60}")
        print(f"processing: {midi_path.name}")
        print("=" * 60)

        try:
            notes = load_notes_from_midi(midi_path)
        except Exception as e:
            print(f"  error loading file: {e}")
            continue

        if len(notes) < 10:
            print(f"  skipping: only {len(notes)} notes")
            continue

        print(f"loaded {len(notes)} notes")

        # Extract onset times
        onset_times = [n["onset_time"] for n in notes]

        # Test IOI calculation
        print("\ncalculating inter-onset intervals...")
        iois = calculate_inter_onset_intervals(onset_times)
        if len(iois) > 0:
            print(f"  IOI stats: min={iois.min():.0f}, max={iois.max():.0f}, median={np.median(iois):.0f}")
        else:
            print("  no IOIs calculated")
            continue

        # Test common IOI detection
        print("\nfinding common IOI values...")
        common_iois = find_common_ioi_values(iois)
        print(f"  common IOIs: {common_iois}")

        # Test density calculation
        print("\ncalculating local density...")
        densities = calculate_local_density(onset_times)
        print(f"  density stats: min={densities.min():.1f}, max={densities.max():.1f}, mean={densities.mean():.1f}")

        # Test grid detection
        print("\ndetecting grid from onsets...")
        grid_times, grid_resolutions = detect_grid_from_onsets(onset_times)
        print(f"  detected {len(grid_times)} grid points")
        if grid_resolutions:
            print(f"  grid resolution range: {min(grid_resolutions)} to {max(grid_resolutions)}")

        # Test quantization
        print("\nquantizing to grid...")
        quant_results = quantize_to_grid(onset_times, grid_times)

        # Analyze quality
        print("\nanalyzing quantization quality...")
        quality = analyze_quantization_quality(quant_results)
        for key, value in quality.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        # Generate visualizations
        name = midi_path.stem
        print(f"\ngenerating visualizations...")
        plot_quantization_analysis(
            notes,
            grid_times,
            quant_results,
            output_dir,
            name=name,
        )

        # Also generate a zoomed view of the first section
        print("generating zoomed view...")
        notes_with_offsets = []
        for note, qr in zip(notes, quant_results):
            note_copy = note.copy()
            note_copy["time_offset"] = qr[2]
            notes_with_offsets.append(note_copy)

        # Zoom to first ~2000 ticks or so
        min_time = min(n["onset_time"] for n in notes)
        plot_piano_roll_with_grid(
            notes_with_offsets,
            grid_times,
            output_dir / f"{name}_zoomed.png",
            time_range=(min_time, min_time + 2000),
            title=f"Piano Roll - {name} (Zoomed)",
        )

    print(f"\n{'='*60}")
    print(f"done! check {output_dir}/ for visualizations.")


if __name__ == "__main__":
    test_quantization_with_real_midi()
