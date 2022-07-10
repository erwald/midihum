from collections import namedtuple
import hashlib
import itertools
from pathlib import Path
from typing import List

from mido import MidiFile, MidiTrack, Message
import numpy as np

Track = namedtuple("Track", ("index", "note_events"))
NoteEvent = namedtuple("NoteEvent", ("index", "time", "type", "note", "velocity"))

def get_note_tracks(midi_file: MidiFile) -> List[Track]:
    """Given a MIDI object, return all the tracks with note events."""
    def _has_note_events(track) -> bool:
        return any(msg.type == "note_on" for msg in track)
    tracks = [
        Track(idx,  get_note_events_for_track(track)) for idx, track in enumerate(midi_file.tracks)
        if _has_note_events(track)]
    assert len(tracks) > 0, tracks
    return tracks

def get_note_events_for_track(track: MidiTrack) -> List[NoteEvent]:
    """Given a MIDI track, return all the note events in it."""
    time_messages = [msg for msg in track if hasattr(msg, "time")]
    cum_times = np.cumsum([msg.time for msg in time_messages])
    return [
        NoteEvent(idx, time, msg.type, msg.note, msg.velocity)
        for (idx, (time, msg)) in enumerate(zip(cum_times, time_messages))
        if msg.type == "note_on" or msg.type == "note_off"]

def get_midi_file_hash(midi_file: MidiFile) -> str:
    concatenated = b""
    for msg in midi_file:
        if not isinstance(msg, Message) or msg.type != "note_on":
            continue
        concatenated += bytes(round(msg.time))
        concatenated += bytes(msg.note)
        concatenated += bytes(msg.velocity)
    return hashlib.md5(concatenated).hexdigest()

def get_midi_filepaths(dir_path: Path) -> List[Path]:
    return list(dir_path.glob("*.mid")) + list(dir_path.glob("*.MID"))
