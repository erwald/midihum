# Much of this code is taken from https://github.com/imalikshake/StyleNet/

import os
import numpy as np
from mido import MidiFile
from midi_utility import *


# Set this to process only files prefixed by the filter string.
FILE_FILTER_PREFIX = None


def validate_data(path, quant):
    '''Creates a folder containing valid MIDI files.

    Arguments:
    path -- Original directory containing untouched midis.
    quant -- Level of quantisation'''

    path_prefix, path_suffix = os.path.split(path)

    # Handle case where a trailing / requires two splits.
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    total_file_count = 0
    processed_count = 0

    base_path_out = os.path.join(path_prefix, path_suffix+'_valid')

    skipped_file_names = []

    for root, _, files in os.walk(path):
        for file in files:
            # Calculate output file path.
            if not os.path.exists(base_path_out):
                os.makedirs(base_path_out)
            output_file_path = os.path.join(base_path_out, file)

            # If the file has already been validated, proceed to the next one.
            if os.path.isfile(output_file_path):
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

    print('\nSkipped {} files:'.format(len(skipped_file_names)))
    for skipped_file_name in skipped_file_names:
        print('\t', skipped_file_name)


def quantize_data(path, quant):
    '''Creates a folder containing the quantised MIDI files.

    Arguments:
    path -- Validated directory containing midis.
    quant -- Level of quantisation'''

    path_prefix, path_suffix = os.path.split(path)

    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    total_file_count = 0
    processed_count = 0

    base_path_out = os.path.join(path_prefix, path_suffix+'_quantized')
    for root, _, files in os.walk(path):
        for file in files:
            # Calculate output file path.
            suffix = root.split(path)[-1]
            out_dir = base_path_out + '/' + suffix
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            output_file_path = os.path.join(out_dir, file)

            # If the file has already been quantized, proceed to the next one.
            if os.path.isfile(output_file_path):
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


def save_data(path, quant):
    '''Creates a folder containing the quantised MIDI files.

    Arguments:
    path -- Quantised directory containing midis.
    quant -- Level of quantisation'''

    path_prefix, path_suffix = os.path.split(path)

    # Handle case where a trailing / requires two splits.
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    array_out = os.path.join(path_prefix, path_suffix+'_inputs')
    velocity_out = os.path.join(path_prefix, path_suffix+'_velocities')

    total_file_count = 0
    processed_count = 0

    for root, _, files in os.walk(path):
        for file in files:
            # Calculate output file paths.
            array_output_path = '{}.npy'.format(os.path.join(array_out, file))
            velocity_output_path = '{}.npy'.format(
                os.path.join(velocity_out, file))

            # If the file has already been saved as data, proceed.
            if (os.path.isfile(array_output_path) and
                    os.path.isfile(velocity_output_path)):
                continue

            is_filtered_out = FILE_FILTER_PREFIX and not file.startswith(
                FILE_FILTER_PREFIX)
            if (file.split('.')[-1] == 'mid' or file.split('.')[-1] == 'MID') and not is_filtered_out:
                total_file_count += 1

                print('Saving', str(file))
                mid = MidiFile(os.path.join(root, file))

                try:
                    array, velocity_array = midi_to_array_one_hot(
                        mid, quant)
                except Exception as e:
                    print("Exception converting MIDI to array:", e)
                    continue

                if not os.path.exists(array_out):
                    os.makedirs(array_out)

                if not os.path.exists(velocity_out):
                    os.makedirs(velocity_out)

                # print(out_dir)

                # print('Saving MIDI array:', array_out)
                # print('Saving velocity array:', velocity_out)

                # print_array(mid, array)
                # raw_input("Press Enter to continue...")

                np.save(array_output_path, array)
                np.save(velocity_output_path, velocity_array)

                processed_count += 1
    print('\nSaved {} files out of {}'.format(
        processed_count, total_file_count))


def load_data(path):
    '''Returns lists of input and output numpy matrices.

    Arguments:
    path -- Quantised directory path.
    quant -- Level of quantisation'''

    names = []
    X_list = []
    Y_list = []
    path_prefix, path_suffix = os.path.split(path)

    # Handle case where a trailing / requires two splits.
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)

    x_path = os.path.join(path_prefix, path_suffix+"_inputs")
    y_path = os.path.join(path_prefix, path_suffix+"_labels")

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
