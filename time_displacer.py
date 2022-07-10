from pathlib import Path
import random

import click
from mido import MidiFile

from midi_utility import get_note_tracks

# TODO: refactor this whole thing ...
def displace(source_path: Path, destination_path: Path):
    """Performs a time displacement on the MIDI file at the given path, saving the result as a new file. What this does
    in practice is it adjust note time values to add some brief pause between notes that would otherwise be played
    simultaneously.
    """
    midi_file = MidiFile(source_path)

    note_event_tracks = [(track_idx, track.note_events) for track_idx, track in get_note_tracks(midi_file)]
    # (391, (145440, 'note_on', 48, 80))
    # (index, (cumulative time, note event, pitch, velocity))
    note_events = [
        (event, event_idx, track_idx) for track_idx, track in note_event_tracks for event_idx, event in track]
    note_events.sort(key=lambda event: event[0][0])

    # Count how many simultaneous note on events there are at each point in time.

    # TODO: make these parameters function arguments
    displacement_max = 60
    displacement_min = 40
    displacement_map = {}
    previous_time = -1
    num_current_events = 0

    import pdb
    pdb.set_trace()

    for event, _, _ in note_events:
        time, msg_type, _, velocity = event

        if msg_type == "note_on" and velocity > 0:
            if previous_time == time:
                num_current_events += 1
            else:
                # If there were more than 1 note at previous time, displace.
                if num_current_events > 1:
                    displacements = [
                        random.randint(displacement_min, displacement_max) for _ in [0] * (num_current_events - 1)] + \
                        [0]
                    click.echo(f"time_displacer got previous time :{previous_time}")
                    click.echo(f"time_displacer got displacements :{displacements}")
                    displacement_map[previous_time] = displacements

                num_current_events = 1
                previous_time = time

    # Need to shuffle order of tracks.
    # Need to take into account separate tracks having separate deltas.

    displacement = 0
    displacement_debt_per_track = [0] * len(note_event_tracks)
    compensation_per_track = [0] * len(note_event_tracks)

    for event, event_idx, track_idx in note_events:
        time, msg_type, _, velocity = event

        displacement = 0

        if msg_type == "note_on" and velocity > 0:
            if time in displacement_map:
                displacement = int(round(displacement_map[time].pop()))
                is_displacing = True
            else:
                is_displacing = False
        elif (msg_type == "note_off" or (msg_type == "note_on" and velocity == 0)):
            pass

        curr_event = midi_file.tracks[track_idx][event_idx]

        if displacement != 0:
            click.echo(
                f"time_displacer track {track_idx} changing {curr_event.time} -> {curr_event.time + displacement}")
            curr_event.time += displacement

            # Create a "debt" for other tracks not affected by this delta.
            displacement_debt_per_track = [d + displacement for d in displacement_debt_per_track]
            displacement_debt_per_track[track_idx] -= displacement

            # Create a small "compensation", so that we speed up the following
            # note(s) a bit (since we have slowed down the tempo by displacing).
            if displacement > 0:
                compensation_per_track = [c + round(displacement / 8) for c in compensation_per_track]

        debt = displacement_debt_per_track[track_idx]
        if debt > 0:
            click.echo(f"time_displacer track {track_idx} changing {curr_event.time} -> {curr_event.time + debt}")
            curr_event.time += debt
            displacement_debt_per_track[track_idx] = 0

        compensation = compensation_per_track[track_idx]
        if compensation > 0 and not is_displacing:
            click.echo(
                f"time_displacer track {track_idx} compensating {curr_event.time} -> {curr_event.time - compensation}")
            time_before = curr_event.time
            curr_event.time = max(time_before - compensation, 0)

            compensation_per_track[track_idx] -= time_before - curr_event.time

    click.echo(f"time_displacer saving time displaced file to {destination_path}")
    midi_file.save(destination_path)

# def create_displacement_map(note_events: List, ):

def get_notes_with_time(note_events, time):
    """Returns all notes that have the given time value."""
    return [event for event in note_events if event[0] == time]
