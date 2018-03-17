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

    for root, _, files in os.walk(path):
        for file in files:
            is_filtered_out = FILE_FILTER_PREFIX and not file.startswith(
                FILE_FILTER_PREFIX)
            if file.split('.')[-1].lower() == 'mid' and not is_filtered_out:
                total_file_count += 1
                print('Validating ' + str(file))
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
                        print('\tTime signature not 4/4. Skipping ...')
                        continue
                else:
                    print(len(time_sig_msgs))
                    print('\tNo time signature. Skipping ...')
                    continue

                mid = quantize(MidiFile(os.path.join(root, file)), quant)
                if not mid:
                    print('Invalid MIDI. Skipping...')
                    continue

                if not os.path.exists(base_path_out):
                    os.makedirs(base_path_out)

                out_file = os.path.join(base_path_out, file)

                print('\tSaving', out_file)
                midi_file.save(out_file)
                processed_count += 1

    print('\nValidated {} files out of {}'.format(
        processed_count, total_file_count))


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
            is_filtered_out = FILE_FILTER_PREFIX and not file.startswith(
                FILE_FILTER_PREFIX)
            if file.split('.')[-1].lower() == 'mid' and not is_filtered_out:
                total_file_count += 1
                mid = quantize(MidiFile(os.path.join(root, file)), quant)
                if not mid:
                    print('Invalid MIDI. Skipping...')
                    continue
                suffix = root.split(path)[-1]
                out_dir = base_path_out + '/' + suffix
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(out_dir, file)

                print('Saving', out_file)
                mid.save(out_file)

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
            is_filtered_out = FILE_FILTER_PREFIX and not file.startswith(
                FILE_FILTER_PREFIX)
            if (file.split('.')[-1] == 'mid' or file.split('.')[-1] == 'MID') and not is_filtered_out:
                total_file_count += 1

                out_array = '{}.npy'.format(os.path.join(array_out, file))
                out_velocity = '{}.npy'.format(
                    os.path.join(velocity_out, file))
                midi_path = os.path.join(root, file)
                midi_file = MidiFile(midi_path)

                print('Saving ' + str(file))
                mid = MidiFile(os.path.join(root, file))

                # mid = quantize(midi_file,
                #                quantization=quant)

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

                np.save(out_array, array)
                np.save(out_velocity, velocity_array)

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
