"""Plotting functions for MIDI analysis and visualization."""

import os
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_config import EDA_DIR, ANALYSIS_DIR, DEFAULT_DPI

sns.set_theme(style="whitegrid", palette="bright")


def plot_data(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Generate EDA plots for training data.

    Args:
        df: DataFrame with training data.
        output_dir: Directory to save plots (default: output/eda/).
    """
    output_dir = output_dir or EDA_DIR
    click.echo(f"plotting data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    categorical_cols = []
    continuous_cols = []
    for col in df.columns:
        blacklist = [
            "_lag",
            "_sma",
            "_ewm",
            "_bollinger",
            "chikou",
            "tenkan",
            "kijun",
            "senkou",
            "cloud",
        ]
        if col in ["name", "midi_event_index", "midi_track_index"] or any(
            s in col for s in blacklist
        ):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in categorical_cols:
        plot = sns.boxplot(x=col, y="velocity", data=df)
        plot.get_figure().savefig(output_dir / f"boxplot_{col}.png")
        plt.clf()

    for col in categorical_cols:
        plot = sns.countplot(x=col, data=df)
        plot.get_figure().savefig(output_dir / f"countplot_{col}.png")
        plt.clf()

    for col in continuous_cols:
        plot = sns.regplot(
            x=col, y="velocity", marker="+", scatter_kws={"alpha": 0.25}, data=df
        )
        plot.get_figure().savefig(output_dir / f"regplot_{col}.png")
        plt.clf()

    for col in continuous_cols:
        plot = sns.histplot(df[col])
        plot.get_figure().savefig(output_dir / f"histplot_{col}.png")
        plt.clf()


def plot_piano_roll_with_grid(
    notes: list,
    grid_times: list,
    output_path: Optional[Path] = None,
    time_range: tuple = None,
    title: str = "Piano Roll with Detected Grid",
    name: str = "piano_roll",
):
    """Plot piano roll showing notes, grid lines, and timing offsets.

    Args:
        notes: List of dicts with keys: onset_time, offset_time, pitch, velocity,
            time_offset (time_offset is optional).
        grid_times: List of grid point times (vertical lines).
        output_path: Path to save the plot (default: output/analysis/{name}.png).
        time_range: Optional (start, end) to limit the time axis.
        title: Plot title.
        name: Used for default filename if output_path not specified.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    if output_path is None:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        output_path = ANALYSIS_DIR / f"{name}.png"

    if not notes:
        click.echo("no notes to plot")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    # filter notes by time range if specified
    if time_range:
        start, end = time_range
        notes = [n for n in notes if n["onset_time"] >= start and n["onset_time"] <= end]
        grid_times = [t for t in grid_times if t >= start and t <= end]

    if not notes:
        click.echo("no notes in specified time range")
        return

    # determine pitch range
    pitches = [n["pitch"] for n in notes]
    min_pitch = min(pitches) - 2
    max_pitch = max(pitches) + 2

    # determine time range
    if time_range:
        min_time, max_time = time_range
    else:
        min_time = min(n["onset_time"] for n in notes)
        max_time = max(n.get("offset_time", n["onset_time"] + 100) for n in notes)

    # draw grid lines first (so notes appear on top)
    for grid_time in grid_times:
        ax.axvline(x=grid_time, color="lightgray", linestyle="-", linewidth=0.5, alpha=0.7)

    # create colormap for offsets (blue = early, red = late, white = on grid)
    # normalize offsets for coloring
    offsets = [n.get("time_offset", 0) for n in notes]
    if offsets and any(o != 0 for o in offsets):
        max_abs_offset = max(abs(o) for o in offsets) or 1
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs_offset, vcenter=0, vmax=max_abs_offset)
        cmap = plt.cm.RdBu_r  # blue = negative (early), red = positive (late)
    else:
        norm = None
        cmap = None

    # draw notes as rectangles
    rectangles = []
    colors = []
    for note in notes:
        onset = note["onset_time"]
        offset = note.get("offset_time", onset + 50)  # default duration if not specified
        pitch = note["pitch"]
        duration = max(offset - onset, 10)  # minimum visible width
        time_offset = note.get("time_offset", 0)

        rect = Rectangle((onset, pitch - 0.4), duration, 0.8)
        rectangles.append(rect)

        if norm and cmap:
            colors.append(cmap(norm(time_offset)))
        else:
            colors.append("steelblue")

    # add all rectangles as a collection
    pc = PatchCollection(rectangles, facecolors=colors, edgecolors="black", linewidths=0.5)
    ax.add_collection(pc)

    # add colorbar if we have offset coloring
    if norm and cmap:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Time Offset (ticks)")
        cbar.ax.axhline(y=0, color="black", linewidth=1)

    # set axis limits and labels
    ax.set_xlim(min_time - 50, max_time + 50)
    ax.set_ylim(min_pitch, max_pitch)
    ax.set_xlabel("Time (MIDI ticks)")
    ax.set_ylabel("Pitch (MIDI note number)")
    ax.set_title(title)

    # add pitch labels for common notes (every octave C)
    pitch_labels = {
        21: "A0", 24: "C1", 36: "C2", 48: "C3", 60: "C4 (middle)",
        72: "C5", 84: "C6", 96: "C7", 108: "C8"
    }
    yticks = [p for p in pitch_labels.keys() if min_pitch <= p <= max_pitch]
    ax.set_yticks(yticks)
    ax.set_yticklabels([pitch_labels[p] for p in yticks])

    plt.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    click.echo(f"saved piano roll to {output_path}")


def plot_quantization_analysis(
    notes: list,
    grid_times: list,
    quantization_results: list,
    output_dir: Optional[Path] = None,
    name: str = "analysis",
):
    """Generate multiple plots analyzing quantization quality.

    Args:
        notes: List of note dicts with onset_time, offset_time, pitch, velocity.
        grid_times: List of grid point times.
        quantization_results: List of (actual_time, quantized_time, offset) tuples.
        output_dir: Directory to save plots (default: output/analysis/).
        name: Prefix for output files.
    """
    output_dir = output_dir or ANALYSIS_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not quantization_results:
        click.echo("no quantization results to analyze")
        return

    offsets = [r[2] for r in quantization_results]

    # add offsets to notes for piano roll
    notes_with_offsets = []
    for note, qr in zip(notes, quantization_results):
        note_copy = note.copy()
        note_copy["time_offset"] = qr[2]
        notes_with_offsets.append(note_copy)

    # 1. piano roll with grid - generate zoomed sections instead of full view
    # (full view at 800k+ ticks makes notes invisible)
    times_arr = np.array([r[0] for r in quantization_results])
    min_time = int(times_arr.min())
    max_time = int(times_arr.max())
    duration = max_time - min_time

    # estimate a reasonable zoom window (about 2000-4000 ticks, ~2-4 measures)
    zoom_window = min(4000, duration // 4)

    # generate 3 zoomed views: start, middle, end
    sections = [
        ("start", min_time, min_time + zoom_window),
        ("middle", min_time + duration // 2 - zoom_window // 2, min_time + duration // 2 + zoom_window // 2),
        ("end", max_time - zoom_window, max_time),
    ]

    for section_name, start, end in sections:
        plot_piano_roll_with_grid(
            notes_with_offsets,
            grid_times,
            output_dir / f"{name}_piano_roll_{section_name}.png",
            time_range=(start, end),
            title=f"Piano Roll - {name} ({section_name})",
        )

    # 2. offset distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    # use integer-aligned bins since offsets are integers
    min_offset = int(np.min(offsets))
    max_offset = int(np.max(offsets))
    bins = range(min_offset, max_offset + 2)  # +2 to include max value
    ax.hist(offsets, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Grid (0 offset)")
    ax.axvline(x=np.mean(offsets), color="green", linestyle="-", linewidth=2, label=f"Mean: {np.mean(offsets):.1f}")
    ax.axvline(x=np.median(offsets), color="orange", linestyle="-", linewidth=2, label=f"Median: {np.median(offsets):.1f}")
    ax.set_xlabel("Time Offset (MIDI ticks)")
    ax.set_ylabel("Count")
    ax.set_title(f"Offset Distribution - {name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / f"{name}_offset_distribution.png", dpi=DEFAULT_DPI)
    plt.close(fig)

    # 3. offset over time (to see if there's drift or patterns)
    fig, ax = plt.subplots(figsize=(12, 6))
    times = [r[0] for r in quantization_results]
    ax.scatter(times, offsets, alpha=0.5, s=10)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (MIDI ticks)")
    ax.set_ylabel("Time Offset (ticks)")
    ax.set_title(f"Offset Over Time - {name}")
    plt.tight_layout()
    fig.savefig(output_dir / f"{name}_offset_over_time.png", dpi=DEFAULT_DPI)
    plt.close(fig)

    click.echo(f"saved quantization analysis to {output_dir}")


def plot_cluster_analysis(
    notes_with_offsets,
    stats: dict,
    output_dir: Optional[Path] = None,
    name: str = "cluster",
):
    """Generate plots for cluster-based quantization analysis.

    Args:
        notes_with_offsets: List of NoteWithOffset objects from quantize_notes_to_clusters().
        stats: Statistics dict from quantize_notes_to_clusters().
        output_dir: Directory to save plots (default: output/analysis/).
        name: Prefix for output filenames.
    """
    output_dir = output_dir or ANALYSIS_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_offsets = np.array([n.time_offset for n in notes_with_offsets])
    multi_offsets = np.array([n.time_offset for n in notes_with_offsets if n.cluster_size > 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot 1: all offsets
    ax1 = axes[0]
    if len(all_offsets) > 0:
        bins = range(int(all_offsets.min()) - 1, int(all_offsets.max()) + 2)
        ax1.hist(all_offsets, bins=bins, edgecolor="black", alpha=0.7)
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Centroid")
        ax1.set_xlabel("Offset from cluster centroid (ticks)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"All notes (n={len(all_offsets)})\nstd={all_offsets.std():.1f}")
        ax1.legend()

    # plot 2: multi-note clusters only (chord spread)
    ax2 = axes[1]
    if len(multi_offsets) > 0:
        bins = range(int(multi_offsets.min()) - 1, int(multi_offsets.max()) + 2)
        ax2.hist(multi_offsets, bins=bins, edgecolor="black", alpha=0.7, color="green")
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Centroid")
        ax2.set_xlabel("Offset from cluster centroid (ticks)")
        ax2.set_ylabel("Count")
        pct = stats["pct_in_multi_clusters"]
        ax2.set_title(
            f"Multi-note clusters only (n={len(multi_offsets)}, {pct:.0f}% of notes)\nstd={multi_offsets.std():.1f}"
        )
        ax2.legend()

    plt.suptitle(f"Cluster-based Quantization - {name}", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f"{name}_cluster_analysis.png", dpi=DEFAULT_DPI)
    plt.close(fig)
    click.echo(f"saved cluster analysis to {output_dir / f'{name}_cluster_analysis.png'}")


def plot_cluster_size_distribution(
    notes_with_offsets,
    output_dir: Optional[Path] = None,
    name: str = "cluster",
):
    """Plot distribution of cluster sizes.

    Args:
        notes_with_offsets: List of NoteWithOffset objects from quantize_notes_to_clusters().
        output_dir: Directory to save plots (default: output/analysis/).
        name: Prefix for output filename.
    """
    output_dir = output_dir or ANALYSIS_DIR
    os.makedirs(output_dir, exist_ok=True)

    cluster_sizes = [n.cluster_size for n in notes_with_offsets]
    unique_sizes, counts = np.unique(cluster_sizes, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(unique_sizes, counts, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cluster size (notes)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster Size Distribution - {name}")
    ax.set_xticks(unique_sizes[:20])  # show first 20 sizes

    plt.tight_layout()
    fig.savefig(output_dir / f"{name}_cluster_sizes.png", dpi=DEFAULT_DPI)
    plt.close(fig)
    click.echo(f"saved cluster size distribution to {output_dir / f'{name}_cluster_sizes.png'}")
