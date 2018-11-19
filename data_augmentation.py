import numpy as np


def augmented_data(midi_data, velocity_data):
    # Get an array of note indices (i.e. 0 for A0, 40 for middle C, etc.).
    notes = np.nonzero(midi_data[:, 0::2])[1]

    # Find the lowest and the highest note of the sample.
    note_min = np.amin(notes)
    note_max = np.amax(notes)

    transposition_intervals = np.arange(-6, 6)

    midi_variations_data = []
    velocity_variations_data = []

    # Create 11 additional versions of the sample, one starting from each of the
    # tones of the twelve-tone scale (via chromatic transposition, so as to keep
    # the intervals of the original sample).
    for transposition in transposition_intervals:
        is_out_of_bounds = (((note_min + transposition) < 0)
                            or ((note_max + transposition) > midi_data.shape[1]))
        if transposition == 0 or is_out_of_bounds:
            continue

        midi_variations_data.append(
            np.roll(midi_data, transposition * 2, axis=1))
        velocity_variations_data.append(
            np.roll(velocity_data, transposition, axis=1))

    return np.array(midi_variations_data), np.array(velocity_variations_data)
