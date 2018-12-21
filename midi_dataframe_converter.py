import numpy as np
import pandas as pd
import os
import pretty_midi

from midi_utility import quantize, get_note_tracks
from mido import Message, MetaMessage, MidiFile, MidiTrack


def midi_files_to_data_frame(midi_filepaths):
    '''Returns a Data Frame containing information from the MIDI files at the
    given path.

    Arguments:
    midi_filepaths -- Directory containing MIDI files.'''

    dfs = []
    processed_count = 0
    skipped_count = 0

    for midi_filepath in midi_filepaths:
        print('Converting {} to data frame'.format(midi_filepath))
        midi_file = MidiFile(midi_filepath)

        try:
            df = midi_file_to_data_frame(midi_file)

            # Create some averages.
            df['mean_sustain'] = df['sustain'].mean()
            df['sustain_adj_by_mean'] = df['sustain'] / df['mean_sustain']

            # Add some metadata.
            df['name'] = os.path.split(midi_file.filename)[-1]  # Song name.
            df['number_of_notes'] = len(df)  # Total number of notes in song.
            df['num_of_notes_adj_by_dur'] = len(
                df) / df['song_duration'][0]  # Total # notes / duration of song.

            # Get time signature. Enable this when we don't use only 4/4.
            if False:
                time_signature_msgs = [msg for msg in midi_file.tracks[0]
                                       if msg.type == 'time_signature']
                assert len(time_signature_msgs) > 0, 'No time signature found'
                assert len(
                    time_signature_msgs) < 2, 'More than one time signature found'
                time_signature = time_signature_msgs[0]
                df['time_signature_num'] = time_signature.numerator
                df['time_signature_den'] = time_signature.denominator

            dfs.append(df)
        except Exception as e:
            print('Exception converting MIDI to data frame:', e)
            skipped_count += 1
            continue

        processed_count += 1

    combined_df = pd.concat(dfs)

    print('\nConverted {} files out of {} to data frames'.format(
        processed_count, processed_count + skipped_count))

    return combined_df


def midi_file_to_data_frame(midi_file):
    '''Returns a the MIDI file represented as a Data Frame.

    Arguments:
    mid -- MIDI object with a 4/4 time signature.'''

    # The (minimum) note duration, represented as 1/2**quantization. So e.g. a
    # quantization value of 4 gives note lengths of multiples of 1/16.
    quantization = 4

    # Quantize the notes to a grid of time steps.
    midi_file = quantize(midi_file, quantization=quantization)

    result = []

    note_event_tracks = [note_events_for_track(track=track,
                                               quantization=quantization)
                         for _, track in get_note_tracks(midi_file)]
    note_events = [event for track in note_event_tracks for event in track]
    note_events.sort(key=lambda event: event[0])

    song_duration = note_events[-1][0]  # Get time of final event.

    currently_playing_notes = []

    for event in note_events:
        time, msg_type, pitch, velocity = event

        if msg_type == 'note_on' and velocity > 0:
            assert not (any(p == pitch for p, _, _ in currently_playing_notes)
                        ), 'Pitch played again before previous one ended'

            # Get interval after the last released note by getting that note and
            # checking the difference between the pitch values.
            if len(result) > 0:
                interval_from_last_released_pitch = pitch - result[-1][2]
            else:
                interval_from_last_released_pitch = 0

            # Get interval after the last pressed note in a similar manner.
            if len(currently_playing_notes) > 0:
                interval_from_last_pressed_pitch = (
                    pitch - currently_playing_notes[-1][0])
            else:
                interval_from_last_pressed_pitch = interval_from_last_released_pitch

            note_on_data = [velocity,
                            time,
                            pitch,
                            pitch % 12,
                            pitch // 12,
                            time / song_duration,
                            -(((time / song_duration) * 2 - 1) ** 2) + 1,
                            interval_from_last_released_pitch,
                            interval_from_last_pressed_pitch,
                            len(currently_playing_notes) + 1,
                            int(len(currently_playing_notes) == 0)]

            currently_playing_notes.append((pitch, time, note_on_data))
        elif (msg_type == 'note_off' or (msg_type == 'note_on' and velocity == 0)):
            assert (any(p == pitch for p, _, _ in currently_playing_notes)
                    ), 'Encountered note off event for pitch that has not been played'

            note_on = _, note_on_time, note_on_data = next(
                x for x in currently_playing_notes if x[0] == pitch)
            currently_playing_notes.remove(note_on)

            sustain_duration = time - note_on_time
            assert sustain_duration > 0, 'Encountered note sustained for a duration of 0'

            note_off_data = [sustain_duration,
                             len(currently_playing_notes)]

            # Add new row to result and sort all rows by note time (2nd column).
            result.append(note_on_data + note_off_data)
            result.sort(key=lambda row: row[1])

    df = pd.DataFrame(result)
    df.columns = ['velocity', 'time', 'pitch', 'pitch_class', 'octave',
                  'nearness_to_end', 'nearness_to_midpoint',
                  'interval_from_released', 'interval_from_pressed',
                  'num_played_notes_pressed', 'follows_pause', 'sustain',
                  'num_played_notes_released']

    df['song_duration'] = song_duration

    df['time_since_last_pressed'] = (df.time - df.time.shift(1)).fillna(0)
    df['time_since_last_released'] = (
        df.time - (df.time.shift(1) + df.sustain.shift(1))).fillna(0)

    # Calculate some rolling means.
    for col in ['pitch', 'octave', 'sustain']:
        add_rolling_column(df, col)
        add_rolling_column(df, col, is_forward=True)

    # Calculate some rolling sums.
    for col in ['sustain', 'num_played_notes_pressed', 'num_played_notes_released']:
        add_rolling_column(df, col, aggregator=pd.core.window.Rolling.sum)
        add_rolling_column(
            df, col, aggregator=pd.core.window.Rolling.sum, is_forward=True)

    # Calculate lag values (calculated by summing).
    for col in ['interval_from_released', 'interval_from_pressed',
                'time_since_last_pressed', 'time_since_last_released']:
        for i in range(1, 6):
            new_col = '{}_lag_{}'.format(col, i)
            df[new_col] = df[col].rolling(i).sum().fillna(0)

        for i in range(1, 6):
            new_col = '{}_fwd_lag_{}'.format(col, i)
            df[new_col] = df[col][::-1].rolling(i).sum().fillna(0)[::-1]

    # Calculate lag values.
    for col in ['pitch_class', 'octave', 'follows_pause']:
        for i in range(1, 11):
            new_col = '{}_lag_{}'.format(col, i)
            df[new_col] = df[col].shift(i).fillna(method='backfill')

        for i in range(1, 11):
            new_col = '{}_fwd_lag_{}'.format(col, i)
            df[new_col] = df[col][::-
                                  1].shift(i).fillna(method='backfill')[::-1]

    return df


def note_events_for_track(track, quantization):
    '''Takes a track and returns a list of notes in the track, as 
    represented by a tuple of (cumulative) time, note type ('on' or 'off'),
    pitch value and velocity.
    '''
    time_messages = [msg for msg in track if hasattr(msg, 'time')]
    cum_times = np.cumsum([msg.time for msg in time_messages])
    return [(time,
             msg.type,
             msg.note,
             msg.velocity)
            for (time, msg) in zip(cum_times, time_messages)
            if msg.type == 'note_on' or msg.type == 'note_off']


def add_rolling_column(df, col, aggregator=pd.core.window.Rolling.mean, is_forward=False):
    '''Takes a data frame and the name of a column in that data frame, and then
    adds 6 new columns with rolling means/sums (depending on given aggregator
    function) for that column (3 going backward and 3 forward).
    '''
    windows = [3, 5, 20]
    for window in windows:
        if not is_forward:
            new_col = '{}_rolling_{}_{}'.format(
                col, aggregator.__name__, window)
            df[new_col] = aggregator(df[col].rolling(
                window)).fillna(method='backfill')
        else:
            new_col = '{}_fwd_rolling_{}_{}'.format(
                col, aggregator.__name__, window)
            df[new_col] = aggregator(df[col][::-1].rolling(
                window)).fillna(method='backfill')[::-1]
