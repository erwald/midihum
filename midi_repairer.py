import numpy as np
import os
import pretty_midi
from mido import Message, MetaMessage, MidiFile, MidiTrack
from fastai import *

from midi_dataframe_converter import note_events_for_track, get_note_tracks
from directories import *


def repair_midi_files(path, quantization=4):
    midi_data_filepaths = get_files(midi_data_valid_path, ['.mid', '.MID'])

    for midi_data_filepath in midi_data_filepaths:
        # Create filepath at which we want to save the repaired file.
        _, path_suffix = os.path.split(midi_data_filepath)
        output_file_path = os.path.join(
            midi_data_valid_repaired_path, path_suffix)

        # If a repaired version of this file already exists, skip it.
        if os.path.isfile(output_file_path):
            continue

        # Repair the MIDI file..
        repaired_midi = repair_midi_file(midi_data_filepath, quantization)

        # Save the repaired file in a different folder.
        repaired_midi.save(output_file_path)


def repair_midi_file(midi_filepath, quantization):
    '''Loads the MIDI file at the given path, attempts to repair it by adding
    missing note off values (and removing unexpected ones), and then returns the
    result.
    '''
    print('Repairing {}'.format(midi_filepath))

    midi_file = MidiFile(midi_filepath)

    for track in get_note_tracks(midi_file):
        sustains = []
        failing_notes = []

        for event in note_events_for_track(track=track, quantization=quantization):
            time, msg_type, pitch, velocity = event

            # Store note event.
            if msg_type == 'note_on' and velocity > 0:
                failing_notes.append(event)
            elif (msg_type == 'note_off' or (msg_type == 'note_on' and velocity == 0)):
                note_on = next(x for x in failing_notes if x[2] == pitch)
                if note_on:
                    failing_notes.remove(note_on)
                    sustains.append(note_on[0] - time)
            else:
                assert False, 'We got an unexpected note event: {}'.format(
                    event)

        if len(failing_notes) > 0:
            print('Found {} note on events with missing note off values:\n{}'.format(
                len(failing_notes), failing_notes))

            mean_sustain = int(np.ceil(np.mean(sustains)))

            for failing_note in failing_notes:
                new_event = failing_note.copy()
                new_event[0] = failing_note[0] + mean_sustain
                new_event[1] = 'note_off'
                new_event[3] = 0
                track.append(new_event)

    return midi_file
