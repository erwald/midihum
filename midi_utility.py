# Much of this code is taken from https://github.com/imalikshake/StyleNet/

import copy
import pprint
import random
from collections import defaultdict
from math import ceil, floor, log

import matplotlib.pyplot as plt
import mido
import numpy as np
import pretty_midi
from mido import Message, MetaMessage, MidiFile, MidiTrack


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
