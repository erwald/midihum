import os
import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Folders
midi_data_inputs_path = './midi_data_valid_quantized_inputs'
midi_data_velocities_path = './midi_data_valid_quantized_velocities'

# Filename nomenclature:
#   - Name: The name of the song, e.g. 'ZHOU02'.
#   - Filename: The name including file extensions, e.g. 'ZHOU02.mid.npy'.
#   - Filepath: The name including extensions and the full path from the working
#               repo, e.g. './midi_data_valid_quantized_inputs/ZHOU02.mid.npy'.


def load_file(name):
    abs_inputs_path = os.path.join(midi_data_inputs_path, name + '.mid.npy')
    abs_velocities_path = os.path.join(
        midi_data_velocities_path, name + '.mid.npy')
    loaded_inputs = np.load(abs_inputs_path)

    loaded_velocities = np.load(abs_velocities_path)
    loaded_velocities = loaded_velocities

    assert loaded_inputs.shape[0] == loaded_velocities.shape[0]

    return loaded_inputs, loaded_velocities


def get_name_from_filepath(filepath):
    # Get last path component and remove extensions.
    return os.path.split(filepath)[-1].split('.')[0]


def midi_names(name_filter='', exclusion_filter=None):
    pattern = '{}*.mid.npy'.format(name_filter)
    filepaths = glob.glob(os.path.join(midi_data_inputs_path, pattern))
    names = [get_name_from_filepath(fp) for fp in filepaths if (
        not exclusion_filter or exclusion_filter not in fp)]
    return names


def any_midi_filename(names=midi_names()):
    return os.path.splitext(sorted(names)[0])[0]


def ratify_sample(x, y, name):
    print('Ratifying {} ...'.format(name))
    assert np.any(x), 'found all-zero input in: {}'.format(name)
    assert np.all(np.isfinite(x)), 'found nan or inf input in: {}'.format(name)
    assert np.any(y), 'found all-zero output in: {}'.format(name)
    assert np.all(np.isfinite(
        y)), 'found nan or inf output in: {}'.format(name)

    notes = x[:, :176][:, 1::2]
    assert not np.any(np.bitwise_xor(np.where(notes > 0, 1, 0), np.where(
        y > 0, 1, 0))), 'Found a zero velocity for note in {}'.format(name)


def get_train_validate_names(test_size, random_state):
    # Get file names of all MIDI files (excluding augmented versions).
    midi_data_names = midi_names(exclusion_filter='_aug_')

    # Split names into training and validation sets.
    return train_test_split(midi_data_names, test_size=test_size, random_state=random_state)


def load_data(names, get_random_augmentation, ratify_data=False):
    '''Loads the musical performances with the given filenames, and returns sets
    of inputs and labels (notes and resulting velocities).'''

    # N songs of Mn timesteps, each with:
    #   - 176 (= 88 * 2) pitch classes
    #   - 1 stress of beat (strong/weak)
    #   - 2 number of notes played and sustained
    #   - 1 time progression
    #   - 1 average pitch value
    #   - 9 values for chord quality (minor, major, suspended, etc.)
    #
    # Iow, each data point: [Mn, 190]
    x = []

    # N songs of Mn timesteps, each with 88 velocities.
    # Iow, each data point: [Mn, 88]
    y = []

    for name in names:
        if get_random_augmentation:
            name = random.choice(midi_names(name_filter=name))

        loaded_inputs, loaded_velocities = load_file(name)
        x.append(loaded_inputs)
        y.append(loaded_velocities)

        if ratify_data:
            ratify_sample(loaded_inputs, loaded_velocities, name)

    x = np.array(x)
    y = np.array(y)

    return x, y
