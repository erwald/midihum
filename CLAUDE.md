# Claude Code Guidelines for midihum

## Overview

midihum is an ML-based MIDI humanization tool that transforms robotic/quantized MIDI into expressive performances. It uses XGBoost gradient boosted trees trained on ~2,600 competition piano performances from the International Piano-e-Competition.

The tool has two main capabilities:
1. **Velocity humanization** - Predicts natural dynamics (loudness) for each note
2. **Time displacement** - Adds subtle timing variations to quantized notes

## Project Structure

```
midihum/
  main.py                    # CLI entry point (click commands)
  midihum_model.py           # Velocity humanization model
  time_displacement_model.py # Time displacement model
  midi_to_df_conversion.py   # Feature extraction (~400 features)
  quantization.py            # Cluster-based timing analysis
  prepare_midi.py            # Data preparation pipelines
  midi_utility.py            # MIDI file parsing utilities
  plotter.py                 # Visualization functions
  plot_config.py             # Plot output paths and settings
  model_cache/               # Trained models and scalers
  midi_data_repaired_cache/  # Training data (2878 MIDI files)
  dfs/                       # Preprocessed training DataFrames
  output/                    # Generated plots (eda/, analysis/, results/)
  test_output/               # Test visualizations
```

## Key Commands

```shell
# Humanize velocity
python main.py humanize input.mid output.mid

# Apply timing humanization
python main.py time_displace input.mid output.mid --scale 1.0

# Prepare training data
python main.py prepare source_dir/ dest_dir/
python main.py prepare_time_disp source_dir/ dest_dir/
```

## Code Style

### Docstrings (PEP 257)

Follow [PEP 257](https://peps.python.org/pep-0257/) conventions:

```python
def compute_cluster_centroids(clusters: List[List[int]]) -> np.ndarray:
    """Compute the centroid (mean time) of each cluster.

    Args:
        clusters: List of clusters, where each cluster is a list of onset times.

    Returns:
        Array of centroid times for each cluster.
    """
```

Key rules:
- Capitalize first letter of summary line
- End summary with a period
- Use `Args:` and `Returns:` sections (capitalized)
- End each parameter/return description with a period
- Use imperative mood ("Compute..." not "Computes...")

### CLI Output (clig.dev)

Follow [clig.dev](https://clig.dev/) guidelines for CLI interfaces:

```python
# Good - lowercase, terse, no redundant prefixes
click.echo(f"loading model from {path}")
click.echo(f"saved {count} files to {output_dir}")
click.echo("no notes to plot")

# Bad - verbose, redundant module names
click.echo(f"TimeDisplacementModel: Loading model from {path}...")
click.echo(f"Plotter: Successfully saved {count} files to {output_dir}!")
```

Key rules:
- Lowercase messages (no sentence case)
- No trailing periods on short messages
- No redundant module/class name prefixes
- Errors go to stderr: `click.echo(f"error: {msg}", err=True)`
- CLI help text: lowercase, no periods (e.g., `"""convert MIDI files to dataframes"""`)

## Development Guidelines

### Always Use Real MIDI Data

When writing tests, visualizations, or analysis code:
- Use files from `midi_data_repaired_cache/` (real performances)
- Never generate synthetic MIDI data for testing
- Use `random.seed(42)` when sampling files for reproducibility

```python
from pathlib import Path
import random
from midi_utility import get_midi_filepaths

midi_files = get_midi_filepaths(Path("midi_data_repaired_cache"))
random.seed(42)
sample = random.sample(midi_files, min(10, len(midi_files)))
```

### Plotting

Use `plot_config.py` for consistent output paths:

```python
from plot_config import EDA_DIR, ANALYSIS_DIR, get_output_path

# Functions use sensible defaults
plotter.plot_data(df)  # saves to output/eda/
plotter.plot_cluster_analysis(notes, stats)  # saves to output/analysis/

# Or specify custom paths
plotter.plot_data(df, output_dir=Path("custom/"))
```

### Time Displacement Architecture

The time displacement model uses cluster-based quantization to detect "intended" beat positions:

1. Notes within 20 ticks form a cluster (chords played together)
2. The cluster centroid is the "intended" beat position
3. Each note's offset from centroid = expressive timing

About 67% of notes fall in multi-note clusters, providing reliable training targets. Single-note clusters have offset=0 by definition.

Key functions in `quantization.py`:
- `cluster_onsets_by_proximity()` - Groups notes into clusters
- `compute_cluster_centroids()` - Finds cluster center times
- `quantize_notes_to_clusters()` - Main API returning `NoteWithOffset` objects

### Feature Engineering

Features are extracted in `midi_to_df_conversion.py`:
- Pitch and pitch class
- Intervals from previous notes
- Note density and timing context
- Chord analysis (character, size)
- Rolling statistics (SMA, EWM)
- Technical indicators (Ichimoku, MACD-style)

### Model Files

- `model_cache/xgboost_model.json` - Velocity model
- `model_cache/std_scaler.pkl` - Velocity feature scaler
- `model_cache/time_displacement.json` - Time displacement model
- `model_cache/time_displacement_scaler.pkl` - Time displacement scaler

## Testing

```shell
# Unit tests for quantization
python -m pytest test_quantization_unit.py -v

# Integration test with real MIDI
python test_quantization.py
```

## Git Practices

### Atomic Commits

Make small, focused commits that do one thing:

```shell
# Good - one logical change per commit
"Add cluster_onsets_by_proximity function"
"Fix off-by-one error in quantization"
"Update docstrings to PEP 257 style"

# Bad - multiple unrelated changes
"Add clustering, fix bugs, update docs, refactor tests"
```

### Commit Messages

Write descriptive commit messages with a body explaining *why*:

```shell
# Use heredoc for multi-line messages (never just `git commit -m "..."`)
git commit -m "$(cat <<'EOF'
Add cluster-based quantization for time displacement

The cluster centroid method detects "intended" beat positions by
grouping notes within 20 ticks. This provides reliable ground-truth
for training since ~67% of notes fall in multi-note clusters.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

Key rules:
- Use imperative mood ("Add feature" not "Added feature")
- Summary line ~50 chars, no period
- Blank line between summary and body
- Body wrapped at 72 chars
- Explain *why* in body, not just *what*
- Always include a body for non-trivial changes

### What to Commit

- **Do commit**: Source code, tests, documentation, config files
- **Don't commit**: Generated files, large data, credentials, `.mid` files, model caches

Files in `.gitignore`:
- `midi_data*/` - Training data (too large)
- `model_cache/` - Trained models (regenerable)
- `output/`, `test_output/`, `plots/` - Generated visualizations
- `dfs/` - Preprocessed DataFrames
- `*.mid` - MIDI files in project root
