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
  model_cache/               # Trained models and scalers
  midi_data_repaired_cache/  # Training data (2878 MIDI files)
  dfs/                       # Preprocessed training DataFrames
  test_output/               # Generated visualizations
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
