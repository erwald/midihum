# Rachel

Rachel is a neural network for humanizing MIDI -- that is, for taking as input MIDI compositions with constant velocities (flat loudness/dynamics) and producing as output those same compositions with predicted velocity (loudness/dynamics) values for each of the contained notes.

## How does one use this?

Currently I'm reworking this codebase. My intention is to commit and push a trained model, which should make it very easy for anyone to run. But for now it's only in a semi-usable state.

## How does the model look?

The program uses the [fast.ai](https://www.fast.ai/) tabular model (where each row is one note on event; see `rachel_tabular.py`) with a bunch (~1k) of derived features (see `midi_to_df_conversion.py`).

## Whence the name?

I named the project after Rachel Heard, whose recording of Haydn's _Andante con variazioni in F minor_ for Naxos remains unsurpassed among the many recordings of that piece I've heard so far, and after the violinist Rachel Podger. But I would hardly be surprised to learn that there are many more musical Rachels whom this project would be glad to claim as eponyms.
