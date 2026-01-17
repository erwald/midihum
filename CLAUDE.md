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
- `quantization.py` - Grid detection and quantization for time displacement
- `plotter.py` - Visualization functions for analysis
- `midihum_model.py` - The main humanization model
