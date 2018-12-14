import numpy as np
import pandas as pd
import os
import pretty_midi

from midi_utility import quantize, get_note_tracks
from mido import Message, MetaMessage, MidiFile, MidiTrack


def midi_files_to_data_frame(midi_filepaths, quantization):
    '''Returns a Data Frame containing information from the MIDI files at the
    given path.

    Arguments:
    path -- Directory containing MIDI files.
    quantization -- Level of quantisation.'''

    dfs = []
    processed_count = 0
    skipped_count = 0

    for midi_filepath in midi_filepaths[:10]:
        print('Converting {} to data frame'.format(midi_filepath))
        midi_file = MidiFile(midi_filepath)

        try:
            df = midi_file_to_data_frame(midi_file, quantization)
            df.columns = ['pitch', 'pitch_class',
                          'octave', 'velocity', 'velocity_2']
            df['name'] = os.path.split(midi_file.filename)[-1]
            dfs.append(df)
        except Exception as e:
            print("Exception converting MIDI to data frame:", e)
            skipped_count += 1
            continue

        processed_count += 1

    combined_df = pd.concat(dfs)

    print('\nConverted {} files out of {} to data frames'.format(
        processed_count, processed_count + skipped_count))

    return combined_df


def midi_file_to_data_frame(midi_file, quantization):
    '''Returns a the MIDI file represented as a Data Frame.

    Arguments:
    mid -- MIDI object with a 4/4 time signature.
    quantization -- The note duration, represented as 1/2**quantization.'''

    time_sig_msgs = [msg for msg in midi_file.tracks[0]
                     if msg.type == 'time_signature']
    assert len(time_sig_msgs) > 0, 'No time signature found'
    assert len(time_sig_msgs) < 2, 'More than one time signature found'
    # time_sig = time_sig_msgs[0]

    # Quantize the notes to a grid of time steps.
    midi_file = quantize(midi_file, quantization=quantization)

    values = []
    # ticks_per_quarter = midi_file.ticks_per_beat
    for _, track in get_note_tracks(midi_file):
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                values.append([msg.note,
                               msg.note % 12,
                               msg.note // 12,
                               ((msg.velocity / 127.0 - 0.5) * 2),
                               ((msg.velocity / 127.0 - 0.5) * 2)])

    return pd.DataFrame(values)
