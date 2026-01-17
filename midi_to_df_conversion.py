import os
from pathlib import Path
from typing import List, Dict

import click
import numpy as np
import pandas as pd
from mido import MidiFile
from sklearn import preprocessing
from tqdm import tqdm

from midi_utility import get_note_tracks, get_midi_file_hash
from chord_identifier import chord_attributes
from quantization import (
    calculate_local_density,
    quantize_notes_to_clusters,
)


# TODO: parallelize this, so we can take advantage of multiple cores.
def midi_files_to_df(
    midi_filepaths: List[Path],
    skip_suspicious: bool = True,
    include_time_displacement: bool = False,
) -> pd.DataFrame:
    dfs = []
    hashes_to_filenames: Dict[str, str] = {}
    pbar = tqdm(midi_filepaths)
    for midi_filepath in pbar:
        pbar.set_description(f"midi_to_df_conversion converting {midi_filepath} to df")
        midi_file = MidiFile(midi_filepath)

        midi_file_hash = get_midi_file_hash(midi_file)
        if midi_file_hash in hashes_to_filenames:
            tqdm.write(
                f"midi_to_df_conversion skipping {midi_filepath} since an identical file exists "
                f"({hashes_to_filenames[midi_file_hash]})"
            )
            continue
        hashes_to_filenames[midi_file_hash] = midi_filepath

        try:
            df = _midi_file_to_df(midi_file)

            if skip_suspicious and len(df.velocity.unique()) < 25:
                tqdm.write(
                    f"midi_to_df_conversion skipping {midi_filepath} since it had few unique velocity values"
                )
                continue

            df["name"] = os.path.split(midi_file.filename)[-1]
            df = _add_engineered_features(df)

            if include_time_displacement:
                df = _add_time_displacement_features(df)

            assert not np.any(df.index.duplicated()), (midi_filepath, df)

            # reduce size by downcasting float64 and int64 columns
            for column in df:
                if column == "velocity":
                    df[column] = pd.to_numeric(df[column], downcast="float")
                elif df[column].dtype == "float64":
                    df[column] = pd.to_numeric(df[column], downcast="float")
                elif df[column].dtype == "int64":
                    df[column] = pd.to_numeric(df[column], downcast="integer")

            dfs.append(df)
        # TODO: catch more specific exception
        except Exception as e:
            tqdm.write(
                f"midi_to_df_conversion got exception converting midi to df: {e}"
            )
            raise e

    processed_count = len(dfs)
    total_count = len(midi_filepaths)
    click.echo(
        f"midi_to_df_conversion converted {processed_count} files out of {total_count} to dfs"
    )

    if len(dfs) == 0:
        raise ValueError(
            f"midi_to_df_conversion could not convert any of {total_count} files to dataframes"
        )

    return pd.concat(dfs)


def _midi_file_to_df(midi_file) -> pd.DataFrame:
    note_tracks = get_note_tracks(midi_file)
    note_events = [
        (track.index, note_event)
        for track in note_tracks
        for note_event in track.note_events
    ]
    note_events.sort(key=lambda note_event: note_event[1].time)
    song_duration = note_events[-1][1].time

    result = []
    currently_playing_notes = []
    for track_index, event in note_events:
        if event.type == "note_on" and event.velocity > 0:
            # get interval after the last released note by getting that note and checking the difference between the
            # pitch values
            if len(result) > 0:
                interval_from_last_released_pitch = event.note - result[-1][4]
            else:
                interval_from_last_released_pitch = 0

            # get interval after the last pressed note in a similar manner
            if len(currently_playing_notes) > 0:
                interval_from_last_pressed_pitch = (
                    event.note - currently_playing_notes[-1][0]
                )
            else:
                interval_from_last_pressed_pitch = interval_from_last_released_pitch

            # get the average pitch of all notes currently being played
            curr_pitches = [p for p, _, _ in currently_playing_notes] + [event.note]
            average_pitch = np.mean(curr_pitches)

            # add features denoting the quality of chord being played. that means there are six possible values for the
            # "character":
            #
            #   - is it minor?
            #   - is it major?
            #   - is it diminished?
            #   - is it augmented?
            #   - is it suspended?
            #   - or none of the above.
            chord_attrs = chord_attributes(curr_pitches)
            chord_character = (
                chord_attrs[0]
                if chord_attrs is not None and chord_attrs[0] is not None
                else "none"
            )
            # and seven possible values for the number of notes:
            #
            #   - is it a dyad?
            #   - is it a triad?
            #   - is it a seventh?
            #   - is it a ninth?
            #   - is it an eleventh?
            #   - is it a thirteenth?
            #   - or none of the above.
            chord_size = (
                chord_attrs[1]
                if chord_attrs is not None and chord_attrs[1] is not None
                else "none"
            )

            note_on_data = [
                event.velocity,
                event.time,
                track_index,
                event.index,
                event.note,
                str(event.note % 12),
                event.note // 12,
                average_pitch,
                event.time / song_duration,
                -(((event.time / song_duration) * 2 - 1) ** 2) + 1,
                interval_from_last_pressed_pitch,
                interval_from_last_released_pitch,
                len(currently_playing_notes) + 1,
                int(len(currently_playing_notes) == 0),
                chord_character,
                chord_size,
            ]

            currently_playing_notes.append((event.note, event.time, note_on_data))
        elif event.type == "note_off" or (
            event.type == "note_on" and event.velocity == 0
        ):
            if not (any(note == event.note for note, _, _ in currently_playing_notes)):
                # note off-type event for a pitch that isn't being played
                continue

            note_on = _, note_on_time, note_on_data = next(
                x for x in currently_playing_notes if x[0] == event.note
            )
            currently_playing_notes.remove(note_on)

            sustain_duration = event.time - note_on_time

            # if we get a note with a 0 sustain duration, use the duration of the previous note (if there is one)
            if sustain_duration == 0:
                if len(result) > 0:
                    sustain_duration = result[-1][16]
                else:
                    tqdm.write(
                        f"midi_to_df_conversion warning: got first note with 0 duration; defaulting to 25"
                    )
                    sustain_duration = 25.0

            # get the average pitch of all notes currently being played
            curr_pitches = [p for p, _, _ in currently_playing_notes] + [event.note]
            average_pitch = np.mean(curr_pitches)

            note_off_data = [
                sustain_duration,
                len(currently_playing_notes),
                average_pitch,
            ]

            # add new row to result and sort all rows by note time (2nd column)
            result.append(note_on_data + note_off_data)
            result.sort(key=lambda row: row[1])

    skipped_events = len(note_events) - len(result)
    if skipped_events > 0:
        tqdm.write(
            f"midi_to_df_conversion warning: saw {skipped_events} note off events for pitches that hadn't been played"
        )

    df = pd.DataFrame(result)
    df.columns = [
        "velocity",
        "time",
        "midi_track_index",
        "midi_event_index",
        "pitch",
        "pitch_class",
        "octave",
        "avg_pitch_pressed",
        "nearness_to_end",
        "nearness_to_midpoint",
        "interval_from_pressed",
        "interval_from_released",
        "num_played_notes_pressed",
        "follows_pause",
        "chord_character_pressed",
        "chord_size_pressed",
        "sustain",
        "num_played_notes_released",
        "avg_pitch_released",
    ]
    df["song_duration"] = song_duration

    return df


def _add_engineered_features(
    df: pd.DataFrame, with_extra_features: bool = False
) -> pd.DataFrame:
    """Takes a data frame representing one MIDI song and adds a bunch of
    additional features to it.
    """
    # NOTE: it's faster to create each column individually then merge them all together at the end. ("chord_character",
    # "chord_size", "time_since_last_pressed" and "time_since_last_released" are however needed in the df, so we add
    # those to the df directly.)
    new_cols: Dict[str, pd.Series] = {}

    # calculate "true" chord character and size by bunching all samples within 5 time units together and picking the
    # chord character and size of the last of each group for all of them. this makes it so that, if a chord is played
    # with not all notes perfectly at the same time, even the first notes here will get the information of the full
    # chord (hopefully).
    df["chord_character"] = df.groupby(
        np.floor(df.time / 5) * 5
    ).chord_character_pressed.transform("last")
    df["chord_size"] = df.groupby(
        np.floor(df.time / 5) * 5
    ).chord_size_pressed.transform("last")

    # get time elapsed since last note event(s)
    df["time_since_last_pressed"] = (df.time - df.time.shift()).fillna(0)
    df["time_since_last_released"] = (
        df.time - (df.time.shift() + df.sustain.shift())
    ).fillna(0)

    # get time elapsed since various further events. since some of these happen rather rarely (resulting in some very
    # large values), we also normalize.
    for cat in [
        "pitch_class",
        "octave",
        "follows_pause",
        "chord_character",
        "chord_size",
    ]:
        col_name = f"time_since_{cat}"
        col = pd.Series(
            preprocessing.scale(
                (df.time - df.groupby(cat)["time"].shift()).fillna(0).values
            )
        )
        new_cols[col_name] = col
        new_cols[f"log_{col_name}"] = pd.Series(np.log(col + 1))

    # add some abs cols
    for col in ["interval_from_pressed", "interval_from_released"]:
        base = new_cols[col] if col in new_cols else df[col]
        new_cols[f"abs_{col}"] = np.abs(base)

    # add some log cols
    for col in [
        "time_since_chord_character",
        "time_since_chord_size",
        "time_since_follows_pause",
        "time_since_octave",
        "time_since_pitch_class",
    ]:
        base = new_cols[col] if col in new_cols else df[col]
        new_cols[f"log_{col}"] = pd.Series(np.log10(np.abs(base) + 1))
    for col in [
        "sustain",
        "time_since_last_pressed",
        "time_since_last_released",
        "abs_interval_from_pressed",
        "abs_interval_from_released",
    ]:
        base = new_cols[col] if col in new_cols else df[col]
        new_cols[f"log_{col}"] = pd.Series(np.log(np.abs(base) + 1))

    # calculate some simple moving averages
    sma_aggs = {
        "pitch": ["mean", "min", "max", "std"],
        "log_sustain": ["mean", "min", "max", "std"],
        "interval_from_pressed": ["mean", "min", "max", "std"],
        "log_time_since_last_pressed": ["mean", "min", "max", "std"],
        "log_time_since_follows_pause": ["mean", "min", "max", "std"],
    }
    sma_windows = [15, 30, 75]
    for col, funcs in sma_aggs.items():
        base = new_cols[col] if col in new_cols else df[col]
        for window in sma_windows:
            for func in funcs:
                sma = base.rolling(window).agg(func).bfill()
                new_cols[f"{col}_sma_{func}_{window}"] = sma
                fwd_sma = base[::-1].rolling(window).agg(func).bfill()[::-1]
                new_cols[f"{col}_fwd_sma_{func}_{window}"] = fwd_sma

                if col != "follows_pause":
                    new_cols[f"{col}_sma_{func}_{window}_oscillator"] = base - sma
                    new_cols[f"{col}_fwd_sma_{func}_{window}_oscillator"] = (
                        base - fwd_sma
                    )

    # add ichimoku indicators
    for col in [
        "pitch",
        "log_sustain",
        "interval_from_released",
        "interval_from_pressed",
    ]:
        base = new_cols[col] if col in new_cols else df[col]
        tenkan_sen = (base.rolling(9).max() + base.rolling(9).min()).bfill() / 2.0
        kijun_sen = (base.rolling(26).max() + base.rolling(26).min()).bfill() / 2.0
        senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
        senkou_span_b = (base.rolling(52).max() + base.rolling(52).min()).bfill() / 2.0

        new_cols[f"{col}_tenkan_sen"] = tenkan_sen
        new_cols[f"{col}_kijun_sen"] = kijun_sen
        new_cols[f"{col}_senkou_span_a"] = senkou_span_a
        new_cols[f"{col}_senkou_span_b"] = senkou_span_b
        new_cols[f"{col}_chikou_span"] = base.shift(26).bfill()
        new_cols[f"{col}_cloud_is_green"] = senkou_span_a - senkou_span_b

        new_cols[f"{col}_relative_to_tenkan_sen"] = base - tenkan_sen
        new_cols[f"{col}_relative_to_kijun_sen"] = base - kijun_sen
        new_cols[f"{col}_tenkan_sen_relative_to_kijun_sen"] = tenkan_sen - kijun_sen
        new_cols[f"{col}_relative_to_chikou_span"] = base - base.shift(26).bfill()
        new_cols[f"{col}_relative_to_cloud"] = (
            base - (senkou_span_a + senkou_span_b) / 2.0
        )

    if with_extra_features:
        # add percent change columns
        for col in [
            "pitch",
            "log_sustain",
            "num_played_notes_pressed",
            "num_played_notes_released",
            "interval_from_pressed",
            "interval_from_released",
            "log_time_since_last_pressed",
            "log_time_since_last_released",
        ]:
            base = new_cols[col] if col in new_cols else df[col]
            if col == "pitch":
                new_cols[f"{col}_pct_change"] = base.pct_change().fillna(0.0)
            else:
                new_cols[f"{col}_pct_change"] = pd.Series(
                    (np.abs(base) + 1.0).pct_change().fillna(0.0)
                )

    ewm_aggs = {
        "pitch": ["mean", "std"],
        "log_sustain": ["mean", "std"],
        "num_played_notes_pressed": ["mean", "std"],
        "interval_from_pressed": ["mean", "std"],
        "log_abs_interval_from_released": ["mean", "std"],
        "log_time_since_last_pressed": ["mean", "std"],
        "log_time_since_follows_pause": ["mean", "std"],
    }
    for col, funcs in ewm_aggs.items():
        base = new_cols[col] if col in new_cols else df[col]
        for func in funcs:
            for span in [10, 20, 50]:
                new_cols[f"{col}_ewm_{func}_{span}"] = (
                    base.ewm(span=span).agg(func).bfill()
                )
                new_cols[f"{col}_fwd_ewm_{func}_{span}"] = (
                    base[::-1].ewm(span=span).agg(func).bfill()[::-1]
                )

            # actually macd uses ewms with spans 12 and 26 and a signal ewm with span 9. but 2x those works better.
            macd = (
                base.ewm(span=24).agg(func).bfill()
                - base.ewm(span=52).agg(func).bfill()
            )
            new_cols[f"{col}_ewm_{func}_macd"] = macd
            new_cols[f"{col}_ewm_{func}_macd_signal"] = (
                base.ewm(span=18).agg(func).bfill() - macd
            )

    if with_extra_features:
        # calculate lag values (just taking the values of the previous/next rows)
        for col in ["octave", "follows_pause", "chord_character", "chord_size"]:
            for i in range(1, 6):
                new_cols[f"{col}_lag_{i}"] = (
                    df[col].shift(i).bfill().astype(df[col].dtype)
                )
                new_cols[f"{col}_fwd_lag_{i}"] = (
                    df[col][::-1].shift(i).bfill()[::-1].astype(df[col].dtype)
                )

    if with_extra_features:
        # get some aggregate data of the song as a whole
        aggregators = {
            "pitch": ["sum", "mean", "min", "max", "std"],
            "log_sustain": ["sum", "mean", "min", "max", "std"],
            "octave": ["nunique"],
        }
        aggregated = df.agg(aggregators)
        for col, funcs in aggregators.items():
            for func in funcs:
                new_cols[f"{col}_{func}"] = pd.Series([aggregated[col][func]] * len(df))

    if with_extra_features:
        # total number of notes in song
        note_count = pd.Series([len(df)] * len(df))
        new_cols["note_count"] = note_count
        new_cols["note_count_adj_by_dur"] = note_count / df.song_duration[0]

    for name, new_col in new_cols.items():
        if not pd.api.types.is_numeric_dtype(new_col):
            continue
        assert not np.any(np.isnan(new_col)), (name, new_col)
        assert np.all(np.isfinite(new_col)), (name, new_col)

    return pd.concat(
        [df] + [col.rename(name) for name, col in new_cols.items()], axis=1
    )


def _add_time_displacement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time displacement features to a dataframe for training.

    Uses cluster-based quantization: notes within 20 ticks form clusters
    (chords/simultaneous notes), and the cluster centroid represents the
    "intended" beat position. The time_offset is the difference between
    the actual onset and the cluster centroid.

    Args:
        df: dataframe with 'time', 'pitch', 'velocity' columns

    Returns:
        dataframe with added columns:
        - time_offset: target variable (signed, in MIDI ticks)
        - cluster_size: number of notes in this note's cluster
        - position_in_cluster: 0 = earliest note, 1 = second earliest, etc.
        - in_multi_cluster: 1 if cluster_size > 1, else 0
        - local_density: note density around each onset
    """
    if len(df) < 2:
        df["time_offset"] = 0.0
        df["cluster_size"] = 1
        df["position_in_cluster"] = 0
        df["in_multi_cluster"] = 0
        df["local_density"] = 1.0
        return df

    # prepare notes for cluster-based quantization
    notes = [
        {
            "onset_time": int(row.time),
            "pitch": int(row.pitch),
            "velocity": int(row.velocity),
        }
        for row in df.itertuples()
    ]

    # perform cluster-based quantization
    notes_with_offsets, _ = quantize_notes_to_clusters(notes, gap_threshold=20)

    # create a mapping from (onset_time, pitch) to the NoteWithOffset
    # to handle notes at the same time with different pitches
    offset_map = {}
    for nwo in notes_with_offsets:
        key = (nwo.onset_time, nwo.pitch)
        offset_map[key] = nwo

    # extract features in the same order as the dataframe
    time_offsets = []
    cluster_sizes = []
    positions_in_cluster = []

    for row in df.itertuples():
        key = (int(row.time), int(row.pitch))
        if key in offset_map:
            nwo = offset_map[key]
            time_offsets.append(nwo.time_offset)
            cluster_sizes.append(nwo.cluster_size)
            positions_in_cluster.append(nwo.position_in_cluster)
        else:
            # fallback if note not found (shouldn't happen)
            time_offsets.append(0.0)
            cluster_sizes.append(1)
            positions_in_cluster.append(0)

    df["time_offset"] = time_offsets
    df["cluster_size"] = cluster_sizes
    df["position_in_cluster"] = positions_in_cluster
    df["in_multi_cluster"] = (df["cluster_size"] > 1).astype(int)

    # add local density
    onset_times = df["time"].tolist()
    densities = calculate_local_density(onset_times)
    df["local_density"] = densities

    # add log versions for better model performance
    df["log_local_density"] = np.log(df["local_density"] + 1)
    df["log_cluster_size"] = np.log(df["cluster_size"] + 1)

    # add rolling stats for time_offset (useful for detecting patterns)
    for window in [5, 15, 30]:
        df[f"time_offset_sma_{window}"] = (
            df["time_offset"].rolling(window, min_periods=1).mean()
        )
        df[f"time_offset_std_{window}"] = (
            df["time_offset"].rolling(window, min_periods=1).std().fillna(0)
        )

    return df
