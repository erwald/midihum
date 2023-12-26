import os
from pathlib import Path
import traceback
from typing import Tuple, List, Dict

import click
from mido import MidiFile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from midi_to_df_conversion import midi_files_to_df
from midi_utility import get_note_tracks, NoteEvent, get_midi_filepaths
import plotter

_TRAIN_DATA_FILENAME = "train_data.parquet.gzip"
_VALIDATE_DATA_FILENAME = "validate_data.parquet.gzip"


def load_data(data_dir: Path):
    click.echo("prepare_midi loading data")
    train_df = pd.read_parquet(data_dir / _TRAIN_DATA_FILENAME)
    validate_df = pd.read_parquet(data_dir / _VALIDATE_DATA_FILENAME)
    return train_df, validate_df


def get_sorted_velocity_correlations(df: pd.DataFrame) -> List[Tuple[str, float]]:
    correlations: Dict[str, float] = {}
    pbar = tqdm(
        [
            col
            for col in df.columns
            if col not in ["name", "velocity", "midi_track_index", "midi_event_index"]
            and pd.api.types.is_numeric_dtype(df[col])
        ]
    )
    for col in pbar:
        pbar.set_description(f"prepare_midi calculating {col} correlation")
        correlations[col] = df[col].corr(df.velocity)
    return sorted(correlations.items(), key=lambda x: x[1], reverse=True)


def prepare_midi_data(source_dir: Path, destination_dir: Path):
    click.echo("prepare_midi preparing data")

    # TODO: check that user has write privileges on destination dir, and if not, abort and warn.

    # repair midi files (if needed)
    repaired_midi_cache = Path(str(source_dir) + "_repaired_cache")
    repair_midi_files(source_dir, repaired_midi_cache)

    # load repaired midi files, split and convert to dfs
    midi_data_filepaths = get_midi_filepaths(repaired_midi_cache)
    train_filepaths, validate_filepaths = train_test_split(
        midi_data_filepaths, test_size=0.1, random_state=89253
    )
    train_df = midi_files_to_df(midi_filepaths=train_filepaths)
    validate_df = midi_files_to_df(midi_filepaths=validate_filepaths)

    # print some info about the created / loaded training data
    click.echo(f"train shape: {train_df.shape}")
    click.echo(f"train head:\n{train_df.head()}")
    click.echo(f"train tail:\n{train_df.tail()}")
    correlations = get_sorted_velocity_correlations(train_df)
    click.echo("train velocity correlations:")
    for corr in correlations[:25]:
        click.echo(corr)
    click.echo("and negative:")
    for corr in correlations[-25:]:
        click.echo(corr)
    click.echo("and weakest:")
    for corr in sorted(correlations, key=lambda x: abs(x[1]))[:25]:
        click.echo(corr)

    click.echo(f"prepare_midi loaded {len(midi_data_filepaths)} files; saving")
    os.makedirs(destination_dir, exist_ok=True)
    train_df.to_parquet(destination_dir / _TRAIN_DATA_FILENAME)
    validate_df.to_parquet(destination_dir / _VALIDATE_DATA_FILENAME)

    # plot some visualisations of the training set
    plotter.plot_data(train_df.sample(5000), Path("plots"))


def repair_midi_files(source_dir: Path, cache_dir: Path, bust_cache: bool = False):
    os.makedirs(cache_dir, exist_ok=True)
    pbar = tqdm(get_midi_filepaths(source_dir))
    for midi_filepath in pbar:
        pbar.set_description(f"prepare_midi repairing {midi_filepath}")
        _, path_suffix = os.path.split(midi_filepath)
        output_file_path = cache_dir / path_suffix

        # if a repaired version of this file already exists, skip it (unless we're busting the cache)
        if output_file_path.exists() and not bust_cache:
            continue

        try:
            repaired_midi = load_and_repair_midi_file(midi_filepath)
            repaired_midi.save(output_file_path)
        except EOFError:
            tqdm.write(
                f"prepare_midi skipping {midi_filepath} as mido couldn't load the file (EOFError)"
            )


def load_and_repair_midi_file(midi_filepath: Path) -> MidiFile:
    try:
        midi_file = MidiFile(midi_filepath)
    except Exception as e:
        # mido swallows stack traces, so print it here
        traceback.print_exc()
        click.echo(f"prepare_midi got exception loading {midi_filepath}: {e}")
        raise e

    # some midi files have "dangling" note on events -- note ons that don't have corresponding note offs after them.
    # this fixes that by getting the average duration a note is sustained and, for each dangling note on event, adding
    # note off events after that duration.
    for track in get_note_tracks(midi_file):
        sustains = []
        dangling_note_events = []
        for event in track.note_events:
            if event.type == "note_on" and event.velocity > 0:
                dangling_note_events.append(event)
            elif event.type == "note_off" or (
                event.type == "note_on" and event.velocity == 0
            ):
                note_on_event = next(
                    iter([x for x in dangling_note_events if x.note == event.note]),
                    None,
                )
                if note_on_event:
                    dangling_note_events.remove(note_on_event)
                    sustains.append(note_on_event.time - event.time)
            else:
                assert False, event

        if len(dangling_note_events) > 0:
            tqdm.write(
                f"prepare_midi found {len(dangling_note_events)} dangling note(s) on event(s) for {midi_filepath}"
            )
            mean_sustain = int(np.ceil(np.mean(sustains)))
            for dangling_note in dangling_note_events:
                new_event = NoteEvent(
                    dangling_note.index,
                    dangling_note.time + mean_sustain,
                    "note_off",
                    dangling_note.note,
                    0,
                )
                track.note_events.append(new_event)

    return midi_file
