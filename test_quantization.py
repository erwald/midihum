"""
Test script for quantization and visualization using real MIDI data.

Run with: python test_quantization.py
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mido import MidiFile

from quantization import (
    calculate_inter_onset_intervals,
    estimate_base_beat_duration,
    detect_grid_from_onsets,
    quantize_to_grid,
    analyze_quantization_quality,
)
from plotter import plot_piano_roll_with_grid, plot_quantization_analysis
from midi_utility import get_note_tracks, get_midi_filepaths


def load_notes_from_midi(midi_path: Path) -> tuple:
    """
    Load notes from a MIDI file and return them in the format expected by plotting.

    Returns:
        tuple of (notes list, ticks_per_beat)
        notes: list of note dicts with onset_time, offset_time, pitch, velocity
    """
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat
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

    return notes, ticks_per_beat


def plot_tempo_curve(
    onset_times: list,
    local_beats: list,
    output_path: Path,
    name: str = "analysis",
):
    """Plot the detected local tempo curve over time."""
    fig, ax = plt.subplots(figsize=(14, 5))

    times = np.array(onset_times)
    beats = np.array(local_beats)

    # Convert beat duration to BPM (assuming ticks_per_beat relationship)
    # This is approximate - just for visualization
    ax.plot(times, beats, 'b-', alpha=0.7, linewidth=1)
    ax.scatter(times[::10], beats[::10], c='blue', s=10, alpha=0.5)

    ax.set_xlabel("Time (MIDI ticks)")
    ax.set_ylabel("Local beat duration (ticks)")
    ax.set_title(f"Local Tempo Tracking - {name}")
    ax.grid(True, alpha=0.3)

    # Add secondary y-axis showing approximate BPM
    # (assuming 480 ticks/beat at 120 BPM as reference)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved tempo curve to {output_path}")


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
            notes, ticks_per_beat = load_notes_from_midi(midi_path)
        except Exception as e:
            print(f"  error loading file: {e}")
            continue

        if len(notes) < 10:
            print(f"  skipping: only {len(notes)} notes")
            continue

        print(f"loaded {len(notes)} notes (ticks_per_beat={ticks_per_beat})")

        # Extract onset times
        onset_times = [n["onset_time"] for n in notes]

        # Calculate IOIs and estimate base beat
        print("\ncalculating inter-onset intervals...")
        iois = calculate_inter_onset_intervals(onset_times)
        if len(iois) > 0:
            print(f"  IOI stats: min={iois.min():.0f}, max={iois.max():.0f}, median={np.median(iois):.0f}")
        else:
            print("  no IOIs calculated")
            continue

        print("\nestimating base beat duration...")
        base_beat = estimate_base_beat_duration(iois, ticks_per_beat)
        print(f"  base beat: {base_beat} ticks")

        # Detect adaptive grid with local tempo tracking
        print("\ndetecting adaptive grid with local tempo tracking...")
        grid_times, local_beats = detect_grid_from_onsets(
            onset_times,
            ticks_per_beat=ticks_per_beat,
            subdivisions=4,  # sixteenth note grid
            tempo_window=24,
            tempo_smoothing=5.0,
        )
        print(f"  detected {len(grid_times)} grid points")
        if local_beats:
            local_beats_arr = np.array(local_beats)
            print(f"  local beat range: {local_beats_arr.min():.0f} to {local_beats_arr.max():.0f} ticks")
            print(f"  tempo variation: {local_beats_arr.std():.1f} ticks std")

        # Quantize to grid
        print("\nquantizing to adaptive grid...")
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

        # Plot tempo curve
        plot_tempo_curve(
            onset_times,
            local_beats,
            output_dir / f"{name}_tempo.png",
            name=name,
        )

        # Standard quantization analysis plots
        plot_quantization_analysis(
            notes,
            grid_times,
            quant_results,
            output_dir,
            name=name,
        )

        # Generate zoomed view of first section
        print("generating zoomed view...")
        notes_with_offsets = []
        for note, qr in zip(notes, quant_results):
            note_copy = note.copy()
            note_copy["time_offset"] = qr[2]
            notes_with_offsets.append(note_copy)

        # Zoom to first few beats worth of music
        min_time = min(n["onset_time"] for n in notes)
        zoom_duration = base_beat * 8  # ~2 measures
        plot_piano_roll_with_grid(
            notes_with_offsets,
            grid_times,
            output_dir / f"{name}_zoomed.png",
            time_range=(min_time, min_time + zoom_duration),
            title=f"Piano Roll - {name} (Zoomed, ~2 measures)",
        )

    print(f"\n{'='*60}")
    print(f"done! check {output_dir}/ for visualizations.")


if __name__ == "__main__":
    test_quantization_with_real_midi()
