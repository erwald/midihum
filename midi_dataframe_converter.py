import numpy as np
import pandas as pd
import os
from mido import Message, MetaMessage, MidiFile, MidiTrack
from sklearn import metrics, preprocessing

from midi_utility import quantize, get_note_tracks
from chord_identifier import chord_attributes
from tqdm import tqdm


def midi_files_to_data_frame(midi_filepaths):
    '''Returns a Data Frame containing information from the MIDI files at the
    given path.

    Arguments:
    midi_filepaths -- Directory containing MIDI files.'''

    dfs = []
    processed_count = 0
    skipped_count = 0

    pbar = tqdm(midi_filepaths)
    for midi_filepath in pbar:
        pbar.set_description(f'Converting {midi_filepath} to data frame')
        midi_file = MidiFile(midi_filepath)

        try:
            df = midi_file_to_data_frame(midi_file)

            # Add additional features derived from the existing ones.
            add_engineered_features(df)

            # Add the name of the song.
            df['name'] = os.path.split(midi_file.filename)[-1]

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
            tqdm.write(f'Exception converting MIDI to data frame: {e}')
            skipped_count += 1
            continue

        processed_count += 1

    combined_df = pd.concat(dfs)

    total_count = processed_count + skipped_count
    print(
        f'\nConverted {processed_count} files out of {total_count} to data frames')

    return combined_df


def midi_file_to_data_frame(midi_file, quantization=4):
    '''Returns a the MIDI file represented as a Data Frame.

    Arguments:
    mid -- MIDI object with a 4/4 time signature.
    quantization -- The (minimum) note duration, represented as 
        1/2**quantization. So e.g. a quantization value of 4 gives note lengths 
        of multiples of 1/16.
    '''

    # Quantize the notes to a grid of time steps.
    midi_file = quantize(midi_file, quantization=quantization)

    result = []

    note_event_tracks = [(track_idx, note_events_for_track(track=track))
                         for track_idx, track in get_note_tracks(midi_file)]
    note_events = [(event, event_idx, track_idx)
                   for track_idx, track in note_event_tracks for event_idx, event in track]
    note_events.sort(key=lambda event: event[0][0])

    song_duration = note_events[-1][0][0]  # Get time of final event.

    currently_playing_notes = []

    for event, event_idx, track_idx in note_events:
        time, msg_type, pitch, velocity = event

        if msg_type == 'note_on' and velocity > 0:
            # Get interval after the last released note by getting that note and
            # checking the difference between the pitch values.
            if len(result) > 0:
                interval_from_last_released_pitch = pitch - result[-1][4]
            else:
                interval_from_last_released_pitch = 0

            # Get interval after the last pressed note in a similar manner.
            if len(currently_playing_notes) > 0:
                interval_from_last_pressed_pitch = (
                    pitch - currently_playing_notes[-1][0])
            else:
                interval_from_last_pressed_pitch = interval_from_last_released_pitch

            # Get the average pitch of all notes currently being played.
            curr_pitches = [p for p, _, _ in currently_playing_notes] + [pitch]
            average_pitch = np.mean(curr_pitches)

            # Add features denoting the quality of chord being played.
            #
            # That means there are six possible values for the 'character':
            #   - is it minor?
            #   - is it major?
            #   - is it diminished?
            #   - is it augmented?
            #   - is it suspended?
            #   - or none of the above.
            #
            # and seven possible values for the number of notes:
            #   - is it a dyad?
            #   - is it a triad?
            #   - is it a seventh?
            #   - is it a ninth?
            #   - is it an eleventh?
            #   - is it a thirteenth?
            #   - or none of the above.
            chord_attrs = chord_attributes(curr_pitches)
            chord_character = chord_attrs[0] if chord_attrs is not None and chord_attrs[0] is not None else 'none'
            chord_size = chord_attrs[1] if chord_attrs is not None and chord_attrs[1] is not None else 'none'

            note_on_data = [velocity,
                            time,
                            track_idx,
                            event_idx,
                            pitch,
                            pitch % 12,
                            pitch // 12,
                            average_pitch,
                            time / song_duration,
                            -(((time / song_duration) * 2 - 1) ** 2) + 1,
                            interval_from_last_pressed_pitch,
                            interval_from_last_released_pitch,
                            len(currently_playing_notes) + 1,
                            int(len(currently_playing_notes) == 0),
                            chord_character,
                            chord_size]

            currently_playing_notes.append((pitch, time, note_on_data))
        elif (msg_type == 'note_off' or (msg_type == 'note_on' and velocity == 0)):
            if not (any(p == pitch for p, _, _ in currently_playing_notes)):
                tqdm.write(
                    f'Warning: encountered {msg_type} event with velocity {velocity} for pitch that has not been played')
                continue

            note_on = _, note_on_time, note_on_data = next(
                x for x in currently_playing_notes if x[0] == pitch)
            currently_playing_notes.remove(note_on)

            sustain_duration = time - note_on_time

            # If we get a note with a 0 sustain duration, use the duration of
            # the previous note.
            if sustain_duration == 0:
                sustain_duration = result[-1][16]

            # Get the average pitch of all notes currently being played.
            curr_pitches = [p for p, _, _ in currently_playing_notes] + [pitch]
            average_pitch = np.mean(curr_pitches)

            note_off_data = [sustain_duration,
                             len(currently_playing_notes),
                             average_pitch]

            # Add new row to result and sort all rows by note time (2nd column).
            result.append(note_on_data + note_off_data)
            result.sort(key=lambda row: row[1])

    df = pd.DataFrame(result)
    df.columns = ['velocity', 'time', 'midi_track_index', 'midi_event_index',
                  'pitch', 'pitch_class', 'octave', 'avg_pitch_pressed',
                  'nearness_to_end', 'nearness_to_midpoint',
                  'interval_from_pressed', 'interval_from_released',
                  'num_played_notes_pressed', 'follows_pause',
                  'chord_character_pressed', 'chord_size_pressed', 'sustain',
                  'num_played_notes_released', 'avg_pitch_released']

    df['song_duration'] = song_duration

    return df


def note_events_for_track(track):
    '''Takes a track and returns a list of notes in the track, as
    represented by a tuple of (cumulative) time, note type ('on' or 'off'),
    pitch value and velocity.
    '''
    time_messages = [msg for msg in track if hasattr(msg, 'time')]
    cum_times = np.cumsum([msg.time for msg in time_messages])
    return [(idx, (time,
                   msg.type,
                   msg.note,
                   msg.velocity))
            for (idx, (time, msg)) in enumerate(zip(cum_times, time_messages))
            if msg.type == 'note_on' or msg.type == 'note_off']


def add_engineered_features(df):
    '''Takes a data frame representing one MIDI song and adds a bunch of 
    additional features to it.
    '''
    # Calculate 'true' chord character and size by bunching all samples within
    # 5 time units together and picking the chord character and size of the last
    # of each group for all of them.
    #
    # This makes it so that, if a chord is played with not all notes perfectly
    # at the same time, even the first notes here will get the information of
    # the full chord (hopefully).
    df['chord_character'] = df.groupby(
        np.floor(df.time / 5) * 5).chord_character_pressed.transform('last')
    df['chord_size'] = df.groupby(
        np.floor(df.time / 5) * 5).chord_size_pressed.transform('last')

    # Get time elapsed since last note event(s).
    df['time_since_last_pressed'] = (df.time - df.time.shift()).fillna(0)
    df['time_since_last_released'] = (
        df.time - (df.time.shift() + df.sustain.shift())).fillna(0)

    # Get time elapsed since various further events. Since some of these happen
    # rather rarely (resulting in some very large values), we also normalise.
    for cat in ['pitch_class', 'octave', 'follows_pause', 'chord_character', 'chord_size']:
        new_cat = f'time_since_{cat}'
        df[new_cat] = preprocessing.scale(
            (df.time - df.groupby(cat)['time'].shift()).fillna(0).values)

    # Calculate some rolling means.
    rolling_aggs = {'pitch': ['mean', 'min', 'max', 'std'],
                    'sustain': ['mean', 'min', 'max', 'std'],
                    'num_played_notes_pressed': ['mean', 'min', 'max', 'std'],
                    'num_played_notes_released': ['mean', 'min', 'max', 'std'],
                    'interval_from_released': ['mean', 'min', 'max', 'std'],
                    'interval_from_pressed': ['mean', 'min', 'max', 'std'],
                    'time_since_last_pressed': ['mean', 'min', 'max', 'std'],
                    'time_since_last_released': ['mean', 'min', 'max', 'std']}
    for n in [3, 10, 50]:
        rolled = df.rolling(n).agg(rolling_aggs).fillna(method='backfill')
        fwd_rolled = df[::-
                        1].rolling(n).agg(rolling_aggs).fillna(method='backfill')[::-1]
        for col, funcs in rolling_aggs.items():
            for f in funcs:
                df[f'{col}_roll_{f}_{n}'] = rolled[col][f]
                df[f'{col}_fwd_roll_{f}_{n}'] = fwd_rolled[col][f]

    # Calculate lag values (by summing).
    for col in ['interval_from_released', 'interval_from_pressed',
                'time_since_last_pressed', 'time_since_last_released']:
        for i in range(1, 6):
            new_col = f'{col}_lag_{i}'
            df[new_col] = df[col].rolling(i).sum().fillna(0)

            new_fwd_col = f'{col}_fwd_lag_{i}'
            df[new_fwd_col] = df[col][::-1].rolling(i).sum().fillna(0)[::-1]

    # Calculate lag values (just taking the values of the previous/next rows).
    for col in ['pitch_class', 'octave', 'follows_pause', 'chord_character', 'chord_size']:
        for i in range(1, 11):
            new_col = f'{col}_lag_{i}'
            df[new_col] = df[col].shift(i).fillna(
                method='backfill').astype(df[col].dtype)

            new_fwd_col = f'{col}_fwd_lag_{i}'
            df[new_fwd_col] = df[col][::-
                                      1].shift(i).fillna(method='backfill')[::-1].astype(df[col].dtype)

    # Get some aggregate data of the song as a whole.
    aggregators = {'pitch': ['sum', 'mean', 'min', 'max', 'std'],
                   'sustain': ['sum', 'mean', 'min', 'max', 'std'],
                   'pitch_class': ['nunique']}
    aggregated = df.agg(aggregators)
    for col, funcs in aggregators.items():
        for f in funcs:
            df[f'{col}_{f}'] = aggregated[col][f]

    df['sustain_adj_by_mean'] = df.sustain / df.sustain_mean

    # Count occurrences of some categorical or category-like values.
    df['pitch_class_occur_count'] = df.groupby(
        'pitch_class').pitch_class.transform('count')
    df['octave_occur_count'] = df.groupby('octave').octave.transform('count')
    df['chord_character_occur_count'] = df.groupby(
        'chord_character').chord_character.transform('count')
    df['chord_size_occur_count'] = df.groupby(
        'chord_size').chord_size.transform('count')
    df['pause_count'] = df.follows_pause.value_counts()[1] - 1

    # Total number of notes in song.
    df['note_count'] = len(df)

    # Total number of notes divided by duration of song.
    df['note_count_adj_by_dur'] = len(df) / df.song_duration[0]
