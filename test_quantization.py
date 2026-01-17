"""
Test script for cluster-based quantization using real MIDI data.

Run with: python test_quantization.py
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mido import MidiFile

from quantization import (
    quantize_notes_to_clusters,
    cluster_onsets_by_proximity,
    compute_cluster_centroids,
)
from plotter import plot_piano_roll_with_grid, plot_quantization_analysis
from midi_utility import get_note_tracks, get_midi_filepaths


def load_notes_from_midi(midi_path: Path) -> tuple:
    """
    Load notes from a MIDI file.

    Returns:
        tuple of (notes list, ticks_per_beat)
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


def plot_cluster_analysis(notes_with_offsets, stats, output_dir, name):
    """Generate plots for cluster-based quantization analysis."""

    # Separate multi-note and single-note cluster offsets
    all_offsets = np.array([n.time_offset for n in notes_with_offsets])
    multi_offsets = np.array([n.time_offset for n in notes_with_offsets if n.cluster_size > 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All offsets
    ax1 = axes[0]
    if len(all_offsets) > 0:
        bins = range(int(all_offsets.min()) - 1, int(all_offsets.max()) + 2)
        ax1.hist(all_offsets, bins=bins, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Centroid')
        ax1.set_xlabel('Offset from cluster centroid (ticks)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'All notes (n={len(all_offsets)})\nstd={all_offsets.std():.1f}')
        ax1.legend()

    # Plot 2: Multi-note clusters only (chord spread)
    ax2 = axes[1]
    if len(multi_offsets) > 0:
        bins = range(int(multi_offsets.min()) - 1, int(multi_offsets.max()) + 2)
        ax2.hist(multi_offsets, bins=bins, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Centroid')
        ax2.set_xlabel('Offset from cluster centroid (ticks)')
        ax2.set_ylabel('Count')
        pct = stats['pct_in_multi_clusters']
        ax2.set_title(f'Multi-note clusters only (n={len(multi_offsets)}, {pct:.0f}% of notes)\nstd={multi_offsets.std():.1f}')
        ax2.legend()

    plt.suptitle(f'Cluster-based Quantization - {name}', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f'{name}_cluster_analysis.png', dpi=150)
    plt.close(fig)
    print(f"  saved cluster analysis to {output_dir / f'{name}_cluster_analysis.png'}")


def plot_cluster_size_distribution(notes_with_offsets, output_dir, name):
    """Plot distribution of cluster sizes."""
    cluster_sizes = [n.cluster_size for n in notes_with_offsets]
    unique_sizes, counts = np.unique(cluster_sizes, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(unique_sizes, counts, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Cluster size (notes)')
    ax.set_ylabel('Count')
    ax.set_title(f'Cluster Size Distribution - {name}')
    ax.set_xticks(unique_sizes[:20])  # Show first 20 sizes

    plt.tight_layout()
    fig.savefig(output_dir / f'{name}_cluster_sizes.png', dpi=150)
    plt.close(fig)


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
