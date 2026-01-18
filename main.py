from pathlib import Path

import click

from find_duplicate_midi_files import find_duplicate_midi_files
from midi_scraper import scrape_midi_data
from prepare_midi import prepare_midi_data, prepare_time_displacement_data
from midihum_model import MidihumModel
from time_displacement_model import TimeDisplacementModel


@click.group()
def midihum():
    """humanize MIDI files with ML-predicted velocity and timing"""


@midihum.command()
@click.argument("source_dir")
@click.argument("destination_dir")
def prepare(source_dir: str, destination_dir: str):
    """convert MIDI files to training dataframes"""
    assert source_dir != destination_dir, (source_dir, destination_dir)
    prepare_midi_data(Path(source_dir), Path(destination_dir))


@midihum.command()
@click.argument("destination_dir")
def scrape_midi(destination_dir: str):
    """download e-Piano Competition MIDI files"""
    scrape_midi_data(Path(destination_dir))


@midihum.command()
@click.argument("target_dir")
def find_midi_duplicates(target_dir: str):
    """find duplicate MIDI files in a directory"""
    find_duplicate_midi_files(Path(target_dir))


@midihum.command()
@click.argument("source")
@click.argument("destination")
def humanize(source: str, destination: str):
    """predict velocity values for MIDI file"""
    assert source != destination, (source, destination)
    try:
        MidihumModel().humanize(Path(source), Path(destination))
    except Exception as e:
        click.echo(f"error: {e}", err=True)


@midihum.command()
@click.argument("source_dir")
@click.argument("destination_dir")
def prepare_time_disp(source_dir: str, destination_dir: str):
    """convert MIDI files to time displacement training data"""
    assert source_dir != destination_dir, (source_dir, destination_dir)
    prepare_time_displacement_data(Path(source_dir), Path(destination_dir))


@midihum.command()
@click.argument("source")
@click.argument("destination")
@click.option(
    "--scale",
    default=1.0,
    help="displacement scale (1.0 = full, 0.5 = subtle)",
)
def time_displace(source: str, destination: str, scale: float):
    """apply timing humanization to MIDI file"""
    assert source != destination, (source, destination)
    try:
        TimeDisplacementModel().displace(Path(source), Path(destination), scale)
    except FileNotFoundError as e:
        click.echo(f"error: model not found: {e}", err=True)
        click.echo("run 'python main.py prepare_time_disp' first", err=True)
    except Exception as e:
        click.echo(f"error: {e}", err=True)


if __name__ == "__main__":
    midihum()  # pylint: disable=no-value-for-parameter
