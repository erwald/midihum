"""
Test script for cluster-based quantization using real MIDI data.

Run with: python test_quantization.py
"""

import random
from pathlib import Path

import numpy as np
from mido import MidiFile

from quantization import (
    quantize_notes_to_clusters,
    cluster_onsets_by_proximity,
)
from plotter import (
    plot_piano_roll_with_grid,
    plot_cluster_analysis,
    plot_cluster_size_distribution,
)
from plot_config import TEST_OUTPUT_DIR
from midi_utility import get_note_tracks, get_midi_filepaths


def load_notes_from_midi(midi_path: Path) -> tuple:
    """Load notes from a MIDI file.

    Returns:
        Tuple of (notes list, ticks_per_beat).
    """
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat
    tracks = get_note_tracks(midi_file)

    notes = []
    active_notes = {}

    for track in tracks:
        for event in track.note_events:
            key = (track.index, event.note)

            if event.type == "note_on" and event.velocity > 0:
                active_notes[key] = {
                    "onset_time": event.time,
                    "pitch": event.note,
                    "velocity": event.velocity,
                }
            elif event.type == "note_off" or (event.type == "note_on" and event.velocity == 0):
                if key in active_notes:
                    note = active_notes.pop(key)
                    note["offset_time"] = event.time
                    notes.append(note)

    for note in active_notes.values():
        note["offset_time"] = note["onset_time"] + 100
        notes.append(note)

    notes.sort(key=lambda n: n["onset_time"])
    return notes, ticks_per_beat


def test_quantization_with_real_midi():
    """Run cluster-based quantization tests with real MIDI data."""
    midi_dir = Path("midi_data_repaired_cache")

    if not midi_dir.exists():
        print(f"error: {midi_dir} not found")
        return

    midi_files = get_midi_filepaths(midi_dir)
    if not midi_files:
        print(f"error: no MIDI files found in {midi_dir}")
        return

    random.seed(42)
    selected_files = random.sample(midi_files, min(3, len(midi_files)))

    output_dir = TEST_OUTPUT_DIR
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

        # Cluster-based quantization
        print("\nperforming cluster-based quantization...")
        notes_with_offsets, stats = quantize_notes_to_clusters(notes, gap_threshold=20)

        print(f"  clusters: {stats['num_clusters']}")
        print(f"  multi-note clusters: {stats['multi_note_clusters']}")
        print(f"  single-note clusters: {stats['single_note_clusters']}")
        print(f"  notes in multi-clusters: {stats['notes_in_multi_clusters']} ({stats['pct_in_multi_clusters']:.1f}%)")
        print(f"  multi-cluster offset std: {stats['multi_offset_std']:.1f}")
        print(f"  multi-cluster offset range: {stats['multi_offset_range']}")

        # Generate visualizations
        name = midi_path.stem
        print(f"\ngenerating visualizations...")

        # Cluster analysis plots
        plot_cluster_analysis(notes_with_offsets, stats, output_dir, name)
        plot_cluster_size_distribution(notes_with_offsets, output_dir, name)

        # Piano roll with cluster centroids as grid
        onset_times = [n["onset_time"] for n in notes]
        clusters = cluster_onsets_by_proximity(onset_times, gap_threshold=20)
        grid_times = [int(np.mean(c)) for c in clusters]

        # Add offset info to notes for plotting
        notes_for_plot = []
        for n, nwo in zip(sorted(notes, key=lambda x: x["onset_time"]), notes_with_offsets):
            note_copy = n.copy()
            note_copy["time_offset"] = nwo.time_offset
            notes_for_plot.append(note_copy)

        # Zoomed piano roll
        min_time = min(n["onset_time"] for n in notes)
        zoom_duration = ticks_per_beat * 8
        plot_piano_roll_with_grid(
            notes_for_plot,
            grid_times,
            output_dir / f"{name}_piano_roll_zoomed.png",
            time_range=(min_time, min_time + zoom_duration),
            title=f"Piano Roll - {name} (cluster centroids as grid)",
        )

    print(f"\n{'='*60}")
    print(f"done! check {output_dir}/ for visualizations.")


if __name__ == "__main__":
    test_quantization_with_real_midi()
