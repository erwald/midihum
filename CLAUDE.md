# CLAUDE.md

Guidelines for working on this codebase.

## Project Overview

midihum is a command-line tool for humanizing MIDI files - it takes MIDI compositions as input and produces the same compositions with new velocity (loudness/dynamics) values for each note. It uses XGBoost gradient boosted trees with ~400 engineered features, trained on 2.6K competition piano performances.

## Commands

Run via `python main.py <command>`:

- `humanize <source> <destination>` - Humanize a MIDI file (main feature)
- `prepare <source_dir> <destination_dir>` - Convert MIDI data to DataFrames for training
- `scrape_midi <destination_dir>` - Download e-Piano Competition MIDI files
- `find_midi_duplicates <target_dir>` - Find duplicate MIDI files
- `time_displace` - Not implemented (WIP)

## Key Files

- `main.py` - CLI entry point
- `midihum_model.py` - XGBoost model for velocity prediction
- `midi_to_df_conversion.py` - Feature engineering (~400 features)
- `prepare_midi.py` - MIDI file preprocessing and repair
- `midi_utility.py` - MIDI helper functions (NoteEvent, Track namedtuples)
- `midi_scraper.py` - Downloads training data from piano-e-competition.com
- `plotter.py` - Visualization utilities

## Dependencies

Follow these best practices for managing dependencies in `requirements.txt`:

1. **Only add dependencies when truly necessary** - Before adding a new package, consider if the functionality can be achieved with existing dependencies or the standard library.

2. **Remove unused dependencies** - Periodically audit dependencies and remove any that are no longer used.

3. **Don't add transitive dependencies** - If package A already depends on package B, don't add B to requirements.txt unless it's directly imported in the codebase.

4. **Search all file types** - When auditing dependencies, search both `.py` files and `.ipynb` notebooks for imports.

5. **Pin versions** - Use version constraints (e.g., `~= 1.5.0`) to ensure reproducible builds while allowing patch updates.

## Code Style

- Use lowercase for comments
- Prefix log messages with the module name (e.g., `click.echo(f"midi_scraper ...")`)
- Use `tqdm.write()` for output within loops that have progress bars
- Use `click.echo()` for regular CLI output

## Testing

- Tests are in `test_*.py` files
- Run tests with: `python -m pytest`
- Extract pure functions where possible to enable unit testing

## MIDI Concepts

- **Delta time**: MIDI messages store time as offset from previous message, not absolute time
- **Note events**: `note_on` (with velocity > 0) starts a note, `note_off` (or `note_on` with velocity=0) ends it
- **Dangling notes**: `note_on` events without corresponding `note_off` - the repair logic in `prepare_midi.py` fixes these
