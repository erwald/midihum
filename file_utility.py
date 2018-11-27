# Much of this code is taken from https://github.com/imalikshake/StyleNet/

import os
import numpy as np
from mido import MidiFile
from midi_utility import *


# Set this to process only files prefixed by the filter string. This is for
# debugging purposes -- it can be useful then to process only one file (or a
# subset of files) instead of all of them.
FILE_FILTER_PREFIX = None


def validate_data(path, quant, overwrite_existing=False):
    '''Creates a folder containing valid MIDI files.

    Arguments:
    path -- Original directory containing untouched midis.
    quant -- Level of quantisation
    overwrite_existing -- Validates files even if validated copies already exist.'''

    path_prefix, path_suffix = os.path.split(path)

    # Handle case where a trailing / requires two splits.
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    total_file_count = 0
    processed_count = 0

    base_path_out = os.path.join(path_prefix, path_suffix + '_valid')

    skipped_file_names = []

    for root, _, files in os.walk(path):
        for file in files:
            # Calculate output file path.
            if not os.path.exists(base_path_out):
                os.makedirs(base_path_out)
            output_file_path = os.path.join(base_path_out, file)

            # If the file has already been validated, proceed to the next one.
            if os.path.isfile(output_file_path) and not overwrite_existing:
                continue

            is_filtered_out = FILE_FILTER_PREFIX and not file.startswith(
                FILE_FILTER_PREFIX)
            if file.split('.')[-1].lower() == 'mid' and not is_filtered_out:
                total_file_count += 1
                print('Validating', str(file))
                midi_path = os.path.join(root, file)
                try:
                    midi_file = MidiFile(midi_path)
                except (KeyError, IOError, TypeError, IndexError, EOFError, ValueError):
                    print("Bad MIDI.")
                    continue
                time_sig_msgs = [
                    msg for msg in midi_file.tracks[0] if msg.type == 'time_signature']

                if len(time_sig_msgs) == 1:
                    time_sig = time_sig_msgs[0]
                    if not (time_sig.numerator == 4 and time_sig.denominator == 4):
                        skipped_file_names.append(file)
                        print('\tTime signature not 4/4. Skipping ...')
                        continue
                else:
                    skipped_file_names.append(file)
                    print('\tNo time signature. Skipping ...')
                    continue

                mid = quantize(MidiFile(os.path.join(root, file)), quant)
                if not mid:
                    skipped_file_names.append(file)
                    print('Invalid MIDI. Skipping...')
                    continue

                print('\tSaving', output_file_path)
                midi_file.save(output_file_path)
                processed_count += 1

    print('\nValidated {} files out of {}'.format(
        processed_count, total_file_count))

    if len(skipped_file_names) > 0:
        print('\nSkipped {} files:'.format(len(skipped_file_names)))
        for skipped_file_name in skipped_file_names:
            print('\t', skipped_file_name)


def quantize_data(path, quant, overwrite_existing=False):
    '''Creates a folder containing the quantised MIDI files.

    Arguments:
    path -- Validated directory containing midis.
    quant -- Level of quantisation
    overwrite_existing -- Quantizes files even if quantized copies already exist.'''

    path_prefix, path_suffix = os.path.split(path)

    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    total_file_count = 0
    processed_count = 0

    base_path_out = os.path.join(path_prefix, path_suffix + '_quantized')
    for root, _, files in os.walk(path):
        for file in files:
            # Calculate output file path.
            suffix = root.split(path)[-1]
            out_dir = os.path.join(base_path_out, suffix)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            output_file_path = os.path.join(out_dir, file)

            # If the file has already been quantized, proceed to the next one.
            if os.path.isfile(output_file_path) and not overwrite_existing:
                continue

            is_filtered_out = FILE_FILTER_PREFIX and not file.startswith(
                FILE_FILTER_PREFIX)
            if file.split('.')[-1].lower() == 'mid' and not is_filtered_out:
                total_file_count += 1
                print('Quantizing', str(file))

                mid = quantize(MidiFile(os.path.join(root, file)), quant)
                if not mid:
                    print('Invalid MIDI. Skipping...')
                    continue

                print('\tSaving', output_file_path)
                mid.save(output_file_path)

                processed_count += 1

    print('\nQuantized {} files out of {}'.format(
        processed_count, total_file_count))


def save_data(path, quant, overwrite_existing=False):
    '''Creates a folder containing the quantised MIDI files.

    Arguments:
    path -- Quantised directory containing midis.
    quant -- Level of quantisation
    overwrite_existing -- Saves files even if processed already exist.'''

    path_prefix, path_suffix = os.path.split(path)

    # Handle case where a trailing / requires two splits.
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    array_out_dir = os.path.join(path_prefix, path_suffix + '_inputs')
    velocity_out_dir = os.path.join(path_prefix, path_suffix + '_velocities')

    total_file_count = 0
    processed_count = 0

    for root, _, files in os.walk(path):
        for file in files:
            # Calculate output file paths.
            name = file.split('.')[0]
            x_output_filepath = os.path.join(
                array_out_dir, '{}.mid.npy'.format(name))
            y_output_filepath = os.path.join(
                velocity_out_dir, '{}.mid.npy'.format(name))

            # If the file has already been saved as data, proceed.
            if (os.path.isfile(x_output_filepath) and
                    os.path.isfile(y_output_filepath)) and not overwrite_existing:
                continue

            is_filtered_out = FILE_FILTER_PREFIX and not file.startswith(
                FILE_FILTER_PREFIX)
            if (file.split('.')[-1] == 'mid' or file.split('.')[-1] == 'MID') and not is_filtered_out:
                total_file_count += 1

                print('Converting to input data and saving', str(file))
                mid = MidiFile(os.path.join(root, file))

                try:
                    arrays, velocity_arrays = midi_to_array_one_hot(mid, quant)
                except Exception as e:
                    print("Exception converting MIDI to array:", e)
                    continue

                if not os.path.exists(array_out_dir):
                    os.makedirs(array_out_dir)

                if not os.path.exists(velocity_out_dir):
                    os.makedirs(velocity_out_dir)

                # Uncomment this to print the MIDI array in a human-readable
                # format (for debugging purposes).
                # print_array(mid, array)

                # Save first array (the original) with the original paths.
                np.save(x_output_filepath, arrays[0])
                np.save(y_output_filepath, velocity_arrays[0])

                # Save additional arrays (the augmented versions of the first)
                # with special suffixes.
                for idx, (array, velocity_array) in enumerate(zip(arrays[1:], velocity_arrays[1:])):
                    x_aug_output_filepath = os.path.join(
                        array_out_dir, '{}_aug_{}.mid.npy'.format(name, idx + 1))
                    y_aug_output_filepath = os.path.join(
                        velocity_out_dir, '{}_aug_{}.mid.npy'.format(name, idx + 1))
                    np.save(x_aug_output_filepath, array)
                    np.save(y_aug_output_filepath, velocity_array)

                processed_count += 1
    print('\nSaved {} files out of {}'.format(
        processed_count, total_file_count))


def load_data(path):
    '''Returns lists of input and output numpy matrices.

    Arguments:
    path -- Quantised directory path.
    quant -- Level of quantisation'''

    X_list = []
    Y_list = []
    path_prefix, path_suffix = os.path.split(path)

    # Handle case where a trailing / requires two splits.
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    x_path = os.path.join(path_prefix, path_suffix + "_inputs")
    y_path = os.path.join(path_prefix, path_suffix + "_labels")

    for filename in os.listdir(x_path):
        if filename.split('.')[-1] == 'npy':
            abs_path = os.path.join(x_path, filename)
            loaded = np.array(np.load(abs_path))

            X_list.append(loaded)

    for filename in os.listdir(y_path):
        if filename.split('.')[-1] == 'npy':
            abs_path = os.path.join(y_path, filename)
            loaded = np.array(np.load(abs_path))
            Y_list.append(loaded)

    return X_list, Y_list
