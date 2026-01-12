import pytest
from mido import Message

from midi_utility import NoteEvent
from prepare_midi import (
    find_dangling_notes_and_sustains,
    track_to_absolute_times,
    create_note_offs_for_dangling_notes,
    rebuild_track_with_messages,
)


class TestFindDanglingNotesAndSustains:
    def test_no_notes(self):
        dangling, sustains = find_dangling_notes_and_sustains([])
        assert dangling == []
        assert sustains == []

    def test_complete_note_pair(self):
        """a note_on followed by note_off should not be dangling"""
        events = [
            NoteEvent(index=0, time=0, type="note_on", note=60, velocity=100),
            NoteEvent(index=1, time=100, type="note_off", note=60, velocity=0),
        ]
        dangling, sustains = find_dangling_notes_and_sustains(events)
        assert dangling == []
        assert sustains == [100]  # note_off_time - note_on_time = 100 - 0

    def test_dangling_note(self):
        """a note_on without note_off should be dangling"""
        events = [
            NoteEvent(index=0, time=0, type="note_on", note=60, velocity=100),
        ]
        dangling, sustains = find_dangling_notes_and_sustains(events)
        assert len(dangling) == 1
        assert dangling[0].note == 60
        assert sustains == []

    def test_note_on_with_zero_velocity_is_note_off(self):
        """note_on with velocity=0 should be treated as note_off"""
        events = [
            NoteEvent(index=0, time=0, type="note_on", note=60, velocity=100),
            NoteEvent(index=1, time=50, type="note_on", note=60, velocity=0),
        ]
        dangling, sustains = find_dangling_notes_and_sustains(events)
        assert dangling == []
        assert sustains == [50]

    def test_multiple_notes_interleaved(self):
        """multiple notes can be active at once"""
        events = [
            NoteEvent(index=0, time=0, type="note_on", note=60, velocity=100),
            NoteEvent(index=1, time=10, type="note_on", note=64, velocity=100),
            NoteEvent(index=2, time=50, type="note_off", note=60, velocity=0),
            NoteEvent(index=3, time=80, type="note_off", note=64, velocity=0),
        ]
        dangling, sustains = find_dangling_notes_and_sustains(events)
        assert dangling == []
        assert sorted(sustains) == [50, 70]  # 50-0=50, 80-10=70

    def test_mixed_complete_and_dangling(self):
        """some notes complete, some dangling"""
        events = [
            NoteEvent(index=0, time=0, type="note_on", note=60, velocity=100),
            NoteEvent(index=1, time=10, type="note_on", note=64, velocity=100),
            NoteEvent(index=2, time=50, type="note_off", note=60, velocity=0),
            # note 64 never gets a note_off
        ]
        dangling, sustains = find_dangling_notes_and_sustains(events)
        assert len(dangling) == 1
        assert dangling[0].note == 64
        assert sustains == [50]


class TestTrackToAbsoluteTimes:
    def test_empty_track(self):
        result = track_to_absolute_times([])
        assert result == []

    def test_single_message(self):
        msg = Message("note_on", note=60, velocity=100, time=10)
        result = track_to_absolute_times([msg])
        assert len(result) == 1
        assert result[0][0] == 10  # absolute time
        assert result[0][1] is msg

    def test_cumulative_times(self):
        """delta times should accumulate to absolute times"""
        messages = [
            Message("note_on", note=60, velocity=100, time=10),
            Message("note_on", note=64, velocity=100, time=20),
            Message("note_off", note=60, velocity=0, time=30),
        ]
        result = track_to_absolute_times(messages)
        assert [t for t, _ in result] == [10, 30, 60]  # 10, 10+20, 10+20+30


class TestCreateNoteOffsForDanglingNotes:
    def test_empty_list(self):
        result = create_note_offs_for_dangling_notes([], 100)
        assert result == []

    def test_single_dangling_note(self):
        dangling = [NoteEvent(index=0, time=50, type="note_on", note=60, velocity=100)]
        result = create_note_offs_for_dangling_notes(dangling, sustain_duration=100)

        assert len(result) == 1
        abs_time, msg = result[0]
        assert abs_time == 150  # 50 + 100
        assert msg.type == "note_off"
        assert msg.note == 60
        assert msg.velocity == 0

    def test_multiple_dangling_notes(self):
        dangling = [
            NoteEvent(index=0, time=0, type="note_on", note=60, velocity=100),
            NoteEvent(index=1, time=25, type="note_on", note=64, velocity=100),
        ]
        result = create_note_offs_for_dangling_notes(dangling, sustain_duration=50)

        assert len(result) == 2
        assert result[0][0] == 50  # 0 + 50
        assert result[0][1].note == 60
        assert result[1][0] == 75  # 25 + 50
        assert result[1][1].note == 64


class TestRebuildTrackWithMessages:
    def test_sorts_by_absolute_time(self):
        """messages should be sorted by absolute time"""
        track = []
        timed_messages = [
            (30, Message("note_off", note=60, velocity=0, time=999)),
            (10, Message("note_on", note=60, velocity=100, time=999)),
        ]
        rebuild_track_with_messages(track, timed_messages)

        assert len(track) == 2
        assert track[0].type == "note_on"
        assert track[1].type == "note_off"

    def test_recalculates_delta_times(self):
        """delta times should be recalculated from absolute times"""
        track = []
        timed_messages = [
            (10, Message("note_on", note=60, velocity=100, time=999)),
            (30, Message("note_on", note=64, velocity=100, time=999)),
            (60, Message("note_off", note=60, velocity=0, time=999)),
        ]
        rebuild_track_with_messages(track, timed_messages)

        assert track[0].time == 10  # 10 - 0
        assert track[1].time == 20  # 30 - 10
        assert track[2].time == 30  # 60 - 30

    def test_clears_existing_track_contents(self):
        """existing track contents should be replaced"""
        track = [Message("control_change", control=64, value=127, time=0)]
        timed_messages = [
            (10, Message("note_on", note=60, velocity=100, time=0)),
        ]
        rebuild_track_with_messages(track, timed_messages)

        assert len(track) == 1
        assert track[0].type == "note_on"
