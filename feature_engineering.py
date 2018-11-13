import numpy as np
from math import ceil
from utility import replace_nan_with_average
from chord_identifier import chord_attributes


chord_character_one_hot_dict = {'minor': [1, 0, 0, 0, 0],
                                'major': [0, 1, 0, 0, 0],
                                'diminished': [0, 0, 1, 0, 0],
                                'augmented': [0, 0, 0, 1, 0],
                                'suspended': [0, 0, 0, 0, 1]
                                }

chord_size_one_hot_dict = {'dyad': [1, 0, 0, 0],
                           'triad': [0, 1, 0, 0],
                           'seventh': [0, 0, 1, 0],
                           'ninth': [0, 0, 0, 1],
                           'eleventh': [0, 0, 0, 1],
                           'thirteenth': [0, 0, 0, 1],
                           }


def chord_attributes_one_hot(attributes):
    '''Takes a tuple with a chord quality (e.g. ('major', 'seventh')), and 
    returns a one-hot encoded array to be used as features in training set.'''
    if attributes == None:
        return np.zeros(9)

    if attributes[0] is not None:
        character = chord_character_one_hot_dict[attributes[0]]
    else:
        character = np.zeros(5)

    if attributes[1] is not None:
        size = chord_size_one_hot_dict[attributes[1]]
    else:
        size = np.zeros(4)

    return np.concatenate((character, size))


measure_beats_dict = {2: [1, 0],
                      3: [1, 0, 0],
                      4: [1, 0, 0.5, 0],
                      6: [1, 0, 0, 0.5, 0, 0]}


def midi_array_with_engineered_features(midi_array, time_sig):
    '''Takes a MIDI array (containing one-hot encoded MIDI events) and returns
    the array with additional, engineered features concatenated onto it.'''
    # Add feature denoting the stress of each beat; iow, for each timestep,
    # whether we are on a strong (1), on a weak (0) or on a medium-strong (0.5)
    # beat. Iow, for 2/2 this would be 1 0 1 0 1 0 1 0 etc., and for 3/4 it
    # would be 1 0 0 1 0 0 1 ...
    assert time_sig.numerator in measure_beats_dict, 'Unsupported time signature'
    measure_beats = measure_beats_dict[time_sig.numerator]
    number_of_measures = ceil(len(midi_array) / len(measure_beats))
    beats = np.tile(measure_beats, number_of_measures)[
        :len(midi_array)][:, None]
    feature_array = np.hstack((midi_array, beats))

    # Add feature denoting how many new notes are played on each beat. Divide by
    # 20 under the assumption that we can play no more notes at any one time.
    played_notes_avg = (np.sum(midi_array[:, 0::2], axis=1) / 20)[:, None]
    feature_array = np.hstack((feature_array, played_notes_avg))

    # Add feature denoting how many notes are sustained on each beat. Divide by
    # 20 under the assumption that we can play no more notes at any one time.
    sustained_notes_avg = (np.sum(midi_array[:, 1::2], axis=1) / 20)[:, None]
    feature_array = np.hstack((feature_array, sustained_notes_avg))

    # Add feature denoting how at which point of the song we are timewise, from
    # 0 (at the very start of it) to 1 (at the very end of it).
    time = (np.arange(0, len(midi_array)) / (len(midi_array) - 1))[:, None]
    feature_array = np.hstack((feature_array, time))

    # Add a feature denoting, for each timestep, the average pitch value for the
    # notes played or sustained at that point.
    notes = np.maximum(midi_array[:, 0::2], midi_array[:, 1::2])
    pitch_values = [[i for i, is_played in enumerate(
        timestep) if is_played == 1] for timestep in notes]
    pitch_value_avg = [np.average(
        step) / 88 if len(step) > 0 else np.nan for step in pitch_values]
    pitch_value_avg = np.array(pitch_value_avg, dtype=np.float64)[:, None]

    # When no notes are sounded, we have nan. Replace all of those with the
    # average pitch for the whole song.
    pitch_value_avg = replace_nan_with_average(pitch_value_avg)

    feature_array = np.hstack((feature_array, pitch_value_avg))

    # Add features denoting the quality of chord being played at each timestep.
    #
    # That means five fields are one-hot encoded for the 'character':
    #   - is it minor?
    #   - is it major?
    #   - is it diminished?
    #   - is it augmented?
    #   - is it suspended?
    #
    # and four fields are one-hot encoded for the number of notes:
    #   - is it a dyad?
    #   - is it a triad?
    #   - is it a seventh?
    #   - is it a ninth, eleventh or thirteenth?
    chord_quality = [chord_attributes_one_hot(
        chord_attributes(pitches)) for pitches in pitch_values]
    chord_quality = np.reshape(np.array(
        chord_quality, dtype=np.float64), (len(midi_array), len(chord_quality[0])))
    feature_array = np.hstack((feature_array, chord_quality))

    return feature_array
