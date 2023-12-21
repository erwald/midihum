from pathlib import Path
from typing import Dict, List, Tuple

import click
from mido import MidiFile
from tqdm import tqdm

from midi_utility import get_midi_file_hash, get_midi_filepaths


def find_duplicate_midi_files(target_dir: Path):
    hashes_to_filenames: Dict[str, str] = {}
    midi_filepaths = get_midi_filepaths(target_dir)
    pbar = tqdm(midi_filepaths)
    duplicates: List[Tuple[str, str]] = []
    for midi_filepath in pbar:
        midi_file = MidiFile(midi_filepath)  # this is pretty slow, unfortunately
        midi_file_hash = get_midi_file_hash(midi_file)
        pbar.set_description(
            f"find_duplicate_midi_files checking {midi_filepath} ({midi_file_hash})"
        )
        if midi_file_hash in hashes_to_filenames:
            duplicates.append((midi_filepath, hashes_to_filenames[midi_file_hash]))
            continue
        hashes_to_filenames[midi_file_hash] = midi_filepath

    for duplicate1, duplicate2 in duplicates:
        click.echo(f"{duplicate1} - {duplicate2}")
