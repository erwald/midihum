# midihum

midihum (the tool formerly known as rachel) is a neural network for humanizing MIDI -- that is, for taking as input MIDI compositions with constant velocities (flat loudness/dynamics) and producing as output those same compositions with new velocity (loudness/dynamics) values for each of the contained notes.

This tool requires Python 3. It has been tested on macOS Ventura 13.0.1 and Debian GNU/Linux 5.10.178-3.

## How does one use this?

Using midihum is easy. First clone the repository and install dependencies:

```shell
pip install -r requirements.txt
```

Then simply:

```shell
python main.py humanize /path/to/file.midi /path/to/humanized_file.midi
```

## How does the model look?

The program uses the [fast.ai](https://www.fast.ai/) tabular model (where each row is one note on event; see `midihum_tabular.py`) with a bunch (~1K) of derived features (see `midi_to_df_conversion.py`).
