from pathlib import Path

import click

from find_duplicate_midi_files import find_duplicate_midi_files
from midi_scraper import scrape_midi_data
from prepare_midi import prepare_midi_data, prepare_time_displacement_data
from midihum_model import MidihumModel
from time_displacement_model import TimeDisplacementModel


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
def humanize(source: str, destination: str):
    """Humanize MIDI file at SOURCE, writing to DESTINATION."""
    assert source != destination, (source, destination)
    try:
        MidihumModel().humanize(Path(source), Path(destination))
    except Exception as e:
        click.echo(f"midihum could not humanize the given file: {e}")


@midihum.command()
@click.argument("source_dir")
@click.argument("destination_dir")
def prepare_time_disp(source_dir: str, destination_dir: str):
    """Prepare training data for time displacement model.

    SOURCE_DIR should contain MIDI files (expressive performances).
    Outputs training/validation data to DESTINATION_DIR.
    """
    assert source_dir != destination_dir, (source_dir, destination_dir)
    prepare_time_displacement_data(Path(source_dir), Path(destination_dir))


@midihum.command()
@click.argument("source")
@click.argument("destination")
@click.option(
    "--scale",
    default=1.0,
    help="Scale factor for displacement (1.0 = full, 0.5 = subtle)",
)
def time_displace(source: str, destination: str, scale: float):
    """Apply humanistic timing to quantized MIDI file.

    SOURCE should be a quantized MIDI file (from DAW/composition software).
    Predicts and applies timing offsets to make it sound more human.
    """
    assert source != destination, (source, destination)
    try:
        TimeDisplacementModel().displace(Path(source), Path(destination), scale)
    except FileNotFoundError as e:
        click.echo(f"midihum time displacement model not found: {e}")
        click.echo("run 'python main.py prepare_time_disp' first to train the model")
    except Exception as e:
        click.echo(f"midihum could not displace the given file: {e}")


if __name__ == "__main__":
    midihum()  # pylint: disable=no-value-for-parameter
