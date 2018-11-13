# Much of this code is taken from https://github.com/imalikshake/StyleNet/

from collections import defaultdict
import copy
from math import log, floor, ceil
import pprint
import matplotlib.pyplot as plt
import pretty_midi
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
import random
from feature_engineering import midi_array_with_engineered_features

# The MIDI pitches we use.
PITCHES = range(21, 109, 1)
OFFSET = 109-21
PITCH_MAP = {p: i for i, p in enumerate(PITCHES)}


def track_to_array_one_hot(track, ticks_per_quarter, quantization):
    '''Return array representation of a 4/4 time signature MIDI track.

    Normalize the number of time steps in track to a power of 2. Then construct
    a T x N*2 array A (T = number of time steps, N = number of MIDI note 
    numbers) where [A(t,n), A(t, n+1)] is the state of the note number at time 
    step t.

    Arguments:
    track -- MIDI track with a 4/4 time signature.
    quantization -- The note duration, represented as 1/2**quantization.'''

    time_messages = [msg for msg in track if hasattr(msg, 'time')]
    cum_times = np.cumsum([msg.time for msg in time_messages])

    track_len_ticks = cum_times[-1]

    # Extract notes from track.
    notes = [
        (int(time * (2**quantization/4) // (ticks_per_quarter)),
         msg.type, msg.note, msg.velocity)
        for (time, msg) in zip(cum_times, time_messages)
        if msg.type == 'note_on' or msg.type == 'note_off']

    num_steps = int(
        round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))

    # Get position and velocity.
    notes.sort(key=lambda args: (args[0], -args[3]))

    # Lists of note events, iow a list of lists of note on/off events. That is,
    # for each pitch we store a list of note on/off events for that pitch.
    note_on_events = [[] for _ in range(len(PITCHES))]
    note_off_events = [[] for _ in range(len(PITCHES))]

    for note_msg in notes:
        (_, note_type, note_num, velocity) = note_msg
        if note_type == 'note_on' and velocity > 0:
            note_on_events[PITCH_MAP[note_num]].append(note_msg)
        elif (note_type == 'note_off' or (note_type == 'note_on' and velocity == 0)):
            note_off_events[PITCH_MAP[note_num]].append(note_msg)

    # Initialise our resulting arrays with all zeroes.
    midi_array = np.zeros((num_steps, len(PITCHES)*2))
    velocity_array = np.zeros((num_steps, len(PITCHES)))

    # For each pitch ...
    for index in range(len(PITCHES)):
        # ... go through all the pairs of corresponding note on and off values.
        assert len(note_on_events[index]) == len(note_off_events[index])
        for (note_on_msg, note_off_msg) in zip(note_on_events[index], note_off_events[index]):
            (on_position, _, _, on_velocity) = note_on_msg
            (off_position, _, _, _) = note_off_msg

            on_position = min(len(midi_array) - 1, on_position)
            off_position = min(len(midi_array) - 1, off_position)

            # Fill in the values for the note on event.
            midi_array[on_position, 2*index] = 1
            midi_array[on_position, 2*index+1] = 1
            velocity_array[on_position, index] = on_velocity

            # Fill in the sustain values between the note on and off events.
            current_position = off_position
            while current_position > on_position:
                midi_array[current_position, 2*index] = 0
                midi_array[current_position, 2*index+1] = 1
                velocity_array[current_position, index] = on_velocity
                current_position -= 1

    assert len(midi_array) == len(velocity_array)
    return midi_array, velocity_array


def midi_to_array_one_hot(mid, quantization):
    '''Return array representation of a 4/4 time signature MIDI object.

    Arguments:
    mid -- MIDI object with a 4/4 time signature.
    quantization -- The note duration, represented as 1/2**quantization.'''

    time_sig_msgs = [msg for msg in mid.tracks[0]
                     if msg.type == 'time_signature']
    assert len(time_sig_msgs) > 0, 'No time signature found'
    assert len(time_sig_msgs) < 2, 'More than one time signature found'
    time_sig = time_sig_msgs[0]
    assert time_sig.numerator == 4 and time_sig.denominator == 4, 'Not 4/4 time.'

    # Quantize the notes to a grid of time steps.
    mid = quantize(mid, quantization=quantization)

    midi_array = np.array([])
    velocity_array = np.array([])
    ticks_per_quarter = mid.ticks_per_beat
    for _, track in get_note_tracks(mid):
        track_midi_array, track_velocity_array = track_to_array_one_hot(
            track, ticks_per_quarter, quantization)
        assert len(track_midi_array) == len(
            track_velocity_array), 'MIDI and velocity track arrays of different length.'

        # If our accumulated array is empty, just use the current track. Else,
        # merge the accumulated and current array by taking the max values.
        if len(midi_array) == 0:
            midi_array = np.array(track_midi_array)
            velocity_array = np.array(track_velocity_array)
        else:
            midi_array = np.maximum(
                midi_array, np.array(track_midi_array, dtype=int))
            velocity_array = np.maximum(
                velocity_array, np.array(track_velocity_array, dtype=int))

    # Get additional features (derived from the MIDI array and concatenated on
    # top of it).
    feature_array = midi_array_with_engineered_features(midi_array, time_sig)

    assert (not np.isnan(feature_array).any()
            ), 'Feature array should contain real numbers'
    assert np.amax(
        feature_array) <= 1, 'Feature array should not contain numbers over 1'
    assert np.amin(
        feature_array) >= 0, 'Feature array should not contain negative numbers'
    assert(len(feature_array) == len(velocity_array),
           'MIDI and velocity arrays of different length')
    return feature_array.tolist(), velocity_array.tolist()


def print_array(mid, array, quantization=4):
    '''Print a binary array representing midi notes.'''
    bar = 1
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_slice = ticks_per_beat/2**quantization

    bars = [x*ticks_per_slice % ticks_per_beat for x in range(0, len(array))]

    res = ''
    for i, slice in enumerate(array):
        for pitch in slice:
            if pitch > 0:
                res += str(int(pitch))
            else:
                res += '-'
        if bars[i] == 0:
            res += str(bar)
            bar += 1
        res += '\n'

    # Take out the last newline
    print(res[:-1])


def get_note_tracks(mid):
    '''Given a MIDI object, return all the tracks with note events.'''

    tracks = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on':
                tracks.append((i, track))
                break

    assert len(
        tracks) > 0, 'MIDI object does not contain any tracks with note messages.'
    return tracks


def quantize_tick(tick, ticks_per_quarter, quantization):
    '''Quantize the timestamp or tick.

    Arguments:
    tick -- An integer timestamp
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as 1/2**quantization
    '''
    assert (ticks_per_quarter * 4) % 2 ** quantization == 0, \
        'Quantization too fine. Ticks per quantum must be an integer.'
    ticks_per_quantum = (ticks_per_quarter * 4) / float(2 ** quantization)
    quantized_ticks = int(
        round(tick / float(ticks_per_quantum)) * ticks_per_quantum)
    return quantized_ticks


def quantize(mid, quantization=5):
    '''Return a midi object whose notes are quantized to
    1/2**quantization notes.

    Arguments:
    mid -- MIDI object
    quantization -- The note duration, represented as
      1/2**quantization.'''

    quantized_mid = copy.deepcopy(mid)

    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    for note_track_idx, note_track in get_note_tracks(mid):
        new_track = quantize_track(
            note_track, mid.ticks_per_beat, quantization)
        if new_track:
            quantized_mid.tracks[note_track_idx] = note_track

    return quantized_mid


def quantize_track(track, ticks_per_quarter, quantization):
    '''Return the differential time stamps of the note_on, note_off, and
    end_of_track events, in order of appearance, with the note_on events
    quantized to the grid given by the quantization.

    Arguments:
    track -- MIDI track containing note event and other messages
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as
      1/2**quantization.'''

    # Message timestamps are represented as differences between
    # consecutive events. Annotate messages with cumulative timestamps.

    # Assume the following structure:
    # [header meta messages] [note messages] [end_of_track message]
    first_note_msg_idx = None
    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break

    cum_msgs = list(zip(
        np.cumsum([msg.time for msg in track[first_note_msg_idx:]]),
        [msg for msg in track[first_note_msg_idx:]]))
    end_of_track_cum_time = cum_msgs[-1][0]

    quantized_track = MidiTrack()
    quantized_track.extend(track[:first_note_msg_idx])

    quantized_msgs = []
    index = 0

    # Iterate through all the MIDI messages searching for 'note on' events.
    for cum_time, msg in cum_msgs:
        if msg.type == 'note_on' and msg.velocity > 0:
            # For each 'note on' event, find the next corresponding 'note off'
            # event for the same note value.
            for other_cum_time, other_msg in cum_msgs[index:]:
                if ((other_msg.type == 'note_off' or
                     (other_msg.type == 'note_on' and other_msg.velocity == 0))
                        and msg.note == other_msg.note):

                    # Quantized 'note on' time.
                    quantized_note_on_cum_time = quantize_tick(
                        cum_time, ticks_per_quarter, quantization)

                    # The cumulative time of a 'note off' event is the quantized
                    # cumulative time of the associated 'note on' plus the
                    # original difference of the unquantized cumulative times.
                    quantized_note_off_cum_time = quantized_note_on_cum_time + \
                        (other_cum_time - cum_time)
                    quantized_msgs.append(
                        (min(end_of_track_cum_time, quantized_note_on_cum_time), msg))
                    quantized_msgs.append(
                        (min(end_of_track_cum_time, quantized_note_off_cum_time), other_msg))

                    # print('Appended', quantized_msgs[-2:])

                    break
        elif msg.type == 'end_of_track':
            quantized_msgs.append((cum_time, msg))

        index += 1

    # Now, sort the quantized messages by (cumulative time, note_type), making
    # sure that note_on events come before note_off events when two event have
    # the same cumulative time. Compute differential times and construct the
    # quantized track messages.
    def sort_msg(args):
        cum_time, msg = args
        return cum_time if (msg.type == 'note_on' and msg.velocity >
                            0) else (cum_time + 0.5)

    quantized_msgs.sort(key=sort_msg)

    diff_times = [quantized_msgs[0][0]] + list(
        np.diff([msg[0] for msg in quantized_msgs]))
    for diff_time, (cum_time, msg) in zip(diff_times, quantized_msgs):
        quantized_track.append(msg.copy(time=diff_time))

    return quantized_track


def stylify(mid, velocity_array, quantization):
    style_mid = copy.deepcopy(mid)

    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    i = 0
    for note_track_idx, note_track in get_note_tracks(mid):
        new_track = stylify_track(
            note_track, mid.ticks_per_beat, velocity_array, quantization)
        style_mid.tracks[note_track_idx] = new_track
        i += 1

    return style_mid


def stylify_track(track, ticks_per_quarter, velocity_array, quantization):
    time_msgs = [msg for msg in track if hasattr(msg, 'time')]

    cum_times = np.cumsum([msg.time for msg in time_msgs])
    track_len_ticks = cum_times[-1]

    num_steps = int(
        round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))

    # Keep track of the last used velocity in case the model failed to predict
    # velocity for a certain note (leaving it at a regrettable 0).
    last_velocity = 64

    cum_index = 0
    for i, time_msg in enumerate(track):
        if hasattr(time_msg, 'time'):
            if time_msg.type == 'note_on' or time_msg.type == 'note_off':
                if time_msg.velocity > 0:
                    pos = int(cum_times[cum_index] *
                              (2**quantization/4) / (ticks_per_quarter))
                    if pos == num_steps:
                        pos = pos - 1
                    if pos > num_steps:
                        continue

                    velocity = velocity_array[pos, PITCH_MAP[time_msg.note]]
                    velocity = velocity * 127  # From (0, 1) to (0, 127).

                    if velocity < 1:
                        print('Warning: predicted velocity for {} was below 1; '.format(time_msg) +
                              'attempting to rectify by using last velocity of {}'.format(last_velocity))
                        velocity = last_velocity

                    last_velocity = velocity

                    track[i].velocity = int(round(velocity))
            cum_index += 1

    return track


def scrub(mid, velocity=10, random=False):
    '''Returns a midi object with one global velocity.

    Sets all velocities to a contant.

    Arguments:
    mid -- MIDI object with a 4/4 time signature
    velocity -- The global velocity'''
    scrubbed_mid = copy.deepcopy(mid)

    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    for note_track_idx, note_track in get_note_tracks(mid):
        if random:
            new_track = scrub_track_random(note_track)
        else:
            new_track = scrub_track(note_track, velocity=10)
        scrubbed_mid.tracks[note_track_idx] = new_track

    return scrubbed_mid


def scrub_track_random(track):
    first_note_msg_idx = None

    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break

    note_msgs = track[first_note_msg_idx:]

    for msg in note_msgs:
        if msg.type == 'note_on' and msg.velocity > 0:
            msg.velocity = random.randint(0, 127)

    return track


def velocity_range(mid):
    '''Returns a count of velocities.

    Counts the range of velocities in a midi object.

    Arguments:
    mid -- MIDI object with a 4/4 time signature'''

    velocities = defaultdict(lambda: 0)
    for _, track in get_note_tracks(mid):
        first_note_msg_idx = None

        for i, msg in enumerate(track):
            if msg.type == 'note_on':
                first_note_msg_idx = i
                break
        note_msgs = track[first_note_msg_idx:]
        for msg in note_msgs:
            if msg.type == 'note_on' and msg.velocity > 0:
                velocities[str(msg.velocity)] += 1

    dynamics = len(velocities.keys())
    if dynamics > 1:
        return dynamics
    else:
        return 0


def scrub_track(track, velocity):
    first_note_msg_idx = None

    for i, msg in enumerate(track):
        if msg.type == 'note_on':
            first_note_msg_idx = i
            break

    note_msgs = track[first_note_msg_idx:]

    for msg in note_msgs:
        if msg.type == 'note_on' and msg.velocity > 0:
            msg.velocity = 10

    return track
