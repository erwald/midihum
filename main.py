from pathlib import Path

import click

from find_duplicate_midi_files import find_duplicate_midi_files
from midi_scraper import scrape_midi_data
from prepare_midi import prepare_midi_data
from midihum_tabular import MidihumTabular
import time_displacer


@click.group()
def midihum():
    """Midihum is a tool for humanizing (that is, determining velocity values of notes for) MIDI files."""


@midihum.command()
@click.argument("source_dir")
@click.argument("destination_dir")
def prepare(source_dir: str, destination_dir: str):
    """Convert MIDI data in SOURCE_DIR to DataFrame and store in DESTINATION_DIR."""
    assert source_dir != destination_dir, (source_dir, destination_dir)
    prepare_midi_data(Path(source_dir), Path(destination_dir))


@midihum.command()
@click.argument("destination_dir")
def scrape_midi(destination_dir: str):
    """Download all e-Piano Competition MIDI files and store in DESTINATION_DIR."""
    scrape_midi_data(Path(destination_dir))


@midihum.command()
@click.argument("target_dir")
def find_midi_duplicates(target_dir: str):
    """Searches target dir for duplicate MIDI files and prints out a list of them. Warning: this is kinda slow."""
    find_duplicate_midi_files(Path(target_dir))


@midihum.command()
@click.argument("source")
@click.argument("destination")
@click.option(
    "--rescale/--no-rescale",
    default=True,
    help="Normalises and scales velocities as a final step.",
)
def humanize(source: str, destination: str, rescale: bool):
    """Humanize MIDI file at SOURCE, writing to DESTINATION."""
    assert source != destination, (source, destination)
    tabular = MidihumTabular(predict_only=True)
    tabular.humanize(Path(source), Path(destination), rescale=rescale)


@midihum.command()
@click.argument("source")
@click.argument("destination")
def time_displace(source: str, destination: str):
    """Displace (jitter) note events of MIDI file at SOURCE, writing to DESTINATION."""
    assert source != destination, (source, destination)
    # time_displacer.displace(Path(source), Path(destination))
    raise NotImplementedError("needs refactoring")


if __name__ == "__main__":
    midihum()  # pylint: disable=no-value-for-parameter
