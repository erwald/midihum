import os
from pathlib import Path

import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="bright")


def plot_data(df: pd.DataFrame, output_dir: Path):
    click.echo("plotter plotting data")
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


def plot_predictions(df: pd.DataFrame, output_dir: Path):
    click.echo("plotter plotting predictions")
    os.makedirs(output_dir, exist_ok=True)

    # relationship between predictions and targets
    for col, err_col in [
        ("prediction", "error"),
        ("adjusted_prediction", "adjusted_error"),
    ]:
        g = sns.FacetGrid(df, col="name", col_wrap=8)
        g.map_dataframe(sns.scatterplot, x="target", y=col, hue=err_col, legend=False)
        g.set(xlim=(-1, 1), ylim=(-1, 1))
        g.savefig(output_dir / f"{col}.png")


def plot_piano_roll_with_grid(
    notes: list,
    grid_times: list,
    output_path: Path,
    time_range: tuple = None,
    title: str = "Piano Roll with Detected Grid",
):
    """
    plot piano roll showing notes, grid lines, and timing offsets.

    args:
        notes: list of dicts with keys: onset_time, offset_time, pitch, velocity, time_offset
               (time_offset is optional, will be calculated if quantization_results provided)
        grid_times: list of grid point times (vertical lines)
        output_path: path to save the plot
        time_range: optional (start, end) to limit the time axis
        title: plot title
    """
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    if not notes:
        click.echo("plotter no notes to plot")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    # filter notes by time range if specified
    if time_range:
        start, end = time_range
        notes = [n for n in notes if n["onset_time"] >= start and n["onset_time"] <= end]
        grid_times = [t for t in grid_times if t >= start and t <= end]

    if not notes:
        click.echo("plotter no notes in specified time range")
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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    click.echo(f"plotter saved piano roll to {output_path}")


def plot_quantization_analysis(
    notes: list,
    grid_times: list,
    quantization_results: list,
    output_dir: Path,
    name: str = "analysis",
):
    """
    generate multiple plots analyzing quantization quality.

    args:
        notes: list of note dicts with onset_time, offset_time, pitch, velocity
        grid_times: list of grid point times
        quantization_results: list of (actual_time, quantized_time, offset) tuples
        output_dir: directory to save plots
        name: prefix for output files
    """
    import numpy as np
    import os

    os.makedirs(output_dir, exist_ok=True)

    if not quantization_results:
        click.echo("plotter no quantization results to analyze")
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
    fig.savefig(output_dir / f"{name}_offset_distribution.png", dpi=150)
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
    fig.savefig(output_dir / f"{name}_offset_over_time.png", dpi=150)
    plt.close(fig)

    click.echo(f"plotter saved quantization analysis to {output_dir}")
