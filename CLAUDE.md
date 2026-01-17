# Claude Code Guidelines for midihum

## Project Overview

midihum is a MIDI humanization tool that takes quantized/robotic MIDI performances and adds human-like expressiveness (timing variations, velocity changes, etc.).

## Important Guidelines

### Always Use Real MIDI Data for Testing and Visualization

When creating test scripts, visualizations, or analysis tools:

- **DO NOT** generate synthetic/artificial MIDI data
- **DO** use real MIDI files from `midi_data_repaired_cache/`
- Use a fixed random seed when selecting files for reproducibility

Real performance data contains the nuances and complexity that synthetic data cannot replicate. Testing with synthetic data can give misleading results about algorithm effectiveness.

Example pattern for loading test data:
```python
from pathlib import Path
import random
from midi_utility import get_midi_filepaths

midi_dir = Path("midi_data_repaired_cache")
midi_files = get_midi_filepaths(midi_dir)
random.seed(42)  # For reproducibility
selected_files = random.sample(midi_files, min(3, len(midi_files)))
```

### Key Directories

- `midi_data_repaired_cache/` - Cached MIDI files from the training dataset (2878 files)
- `test_output/` - Generated visualizations and test outputs

### Key Modules

- `midi_utility.py` - MIDI loading utilities (`get_note_tracks`, `get_midi_filepaths`, `NoteEvent`)
- `quantization.py` - Cluster-based quantization for time displacement
- `plotter.py` - Visualization functions for analysis
- `midihum_model.py` - The main velocity humanization model
- `midi_to_df_conversion.py` - Feature extraction from MIDI to DataFrames

### Time Displacement (Timing Humanization)

The time displacement model uses **cluster-based quantization** to detect where notes "should" be:

1. **Cluster detection**: Notes within 20 ticks of each other form a cluster (chord/simultaneous notes)
2. **Centroid calculation**: The cluster centroid (mean time) represents the "intended" beat position
3. **Offset extraction**: Each note's offset from its cluster centroid is the target for training

Key insight: ~67% of notes are in multi-note clusters, providing reliable ground-truth for timing offsets. The cluster centroid of a chord represents where the chord was "meant" to be played.

Core functions in `quantization.py`:
- `cluster_onsets_by_proximity()` - Groups notes into clusters by temporal proximity
- `compute_cluster_centroids()` - Calculates centroid of each cluster
- `quantize_notes_to_clusters()` - Main API that returns `NoteWithOffset` objects with timing data

Training files:
- `xgboost_train_time_displacement.ipynb` - Training notebook for the time displacement model
- Model saved to `model_cache/time_displacement.json`
