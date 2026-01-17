"""
time displacement model for predicting humanistic timing offsets.

similar to MidihumModel (velocity prediction), but predicts time_offset
instead of velocity. applies timing variations to make quantized MIDI
sound more human.
"""

from pathlib import Path
from typing import List, Tuple

import click
import mido
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import sklearn

from midi_to_df_conversion import midi_files_to_df
from prepare_midi import rebuild_track_with_messages


class TimeDisplacementModel:
    """an XGBoost model for predicting timing offsets of MIDI notes."""

    model_cache_path = Path("model_cache")
    model_path = Path(f"{model_cache_path}/time_displacement.json")
    scaler_path = Path(f"{model_cache_path}/time_displacement_scaler.pkl")

    def __init__(self):
        if self.model_path.exists() and self.scaler_path.exists():
            click.echo(
                f"time_displacement_model loading model from {self.model_path}"
            )
            self.model = xgb.XGBRegressor(
                booster="gbtree",
                max_depth=6,
                learning_rate=0.05,
                n_estimators=1500,
                gamma=0.1,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.6,
                reg_alpha=0.2,
                reg_lambda=0.4,
                n_jobs=8,
                enable_categorical=True,
            )
            self.model.load_model(self.model_path)
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Model files not found at {self.model_path} or {self.scaler_path}"
            )

    @staticmethod
    def _get_column_names_from_df(
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str], List[str]]:
        """get column names split by type, excluding non-feature columns."""
        columns_to_skip = [
            "velocity",
            "time",
            "time_offset",
            "midi_track_index",
            "midi_event_index",
            "name",
        ]
        category_names = [
            col
            for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col]) and col not in columns_to_skip
        ]
        continuous_names = [
            col for col in df.columns if col not in category_names + columns_to_skip
        ]
        out_names = ["time_offset"]
        return (category_names, continuous_names, out_names)

    def add_time_offsets_to_df(
        self,
        df: pd.DataFrame,
        scale_factor: float = 1.0,
    ) -> pd.DataFrame:
        """
        predict time offsets and add them to the dataframe.

        args:
            df: dataframe with MIDI features (from midi_files_to_df)
            scale_factor: multiply predicted offsets by this (1.0 = full humanization)

        returns:
            dataframe with 'predicted_offset' column added
        """
        cat_names, cont_names, out_names = self._get_column_names_from_df(df)
        for col in cat_names:
            df[col] = df[col].astype("category")

        # for inference, we don't have time_offset - need to handle this
        has_target = "time_offset" in df.columns
        if has_target:
            # standardize including output (training mode)
            df[cont_names + out_names] = self.scaler.transform(df[cont_names + out_names])
            features_df = df.drop(out_names, axis=1)
        else:
            # inference mode - only standardize features
            # need to add dummy time_offset column for scaler
            df["time_offset"] = 0
            df[cont_names + out_names] = self.scaler.transform(df[cont_names + out_names])
            features_df = df.drop(out_names, axis=1)

        # make predictions
        features_for_model = features_df.drop(
            ["midi_track_index", "midi_event_index", "name"], axis=1, errors="ignore"
        )
        predictions = self.model.predict(features_for_model)

        # rescale predictions back to original scale
        # get the time_offset column index in the scaler
        time_offset_idx = cont_names.index("time_offset") if "time_offset" in cont_names else -1
        if time_offset_idx >= 0:
            std = np.sqrt(self.scaler.var_[time_offset_idx])
            mean = self.scaler.mean_[time_offset_idx]
            predictions = predictions * std + mean

        df["predicted_offset"] = (predictions * scale_factor).astype(int)

        click.echo(
            f"time_displacement_model inferred {len(df)} offsets with "
            f"mean {np.mean(df.predicted_offset):.1f} and std {np.std(df.predicted_offset):.1f}"
        )
        return df

    def displace(
        self,
        source_path: Path,
        destination_path: Path,
        scale_factor: float = 1.0,
    ) -> List[int]:
        """
        apply time displacement to a MIDI file.

        args:
            source_path: path to input MIDI file (should be quantized)
            destination_path: path to save humanized output
            scale_factor: how much to apply (1.0 = full, 0.5 = subtle)

        returns:
            list of time offsets applied to each note
        """
        click.echo(f"time_displacement_model displacing {source_path}")

        # convert MIDI to dataframe with features
        df = midi_files_to_df(
            midi_filepaths=[source_path],
            skip_suspicious=False,
            include_time_displacement=False,  # inference doesn't need grid detection
        ).copy()

        # add predicted time offsets
        df = self.add_time_offsets_to_df(df, scale_factor)

        # load MIDI file and apply offsets
        midi_file = mido.MidiFile(source_path)
        offsets = df["predicted_offset"].tolist()

        # group notes by track and apply offsets
        # we need to work with absolute times, apply offsets, then rebuild
        for track_idx, track in enumerate(midi_file.tracks):
            # get notes for this track
            track_df = df[df["midi_track_index"] == track_idx]
            if len(track_df) == 0:
                continue

            # convert track to absolute times
            timed_messages = []
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                timed_messages.append((abs_time, msg.copy()))

            # apply offsets to note_on events
            for _, row in track_df.iterrows():
                event_idx = row["midi_event_index"]
                offset = row["predicted_offset"]

                # find the message and apply offset
                for i, (msg_time, msg) in enumerate(timed_messages):
                    if hasattr(msg, "note") and msg.type == "note_on":
                        # check if this is the right event by index
                        if i == event_idx:
                            new_time = max(0, msg_time + offset)
                            timed_messages[i] = (new_time, msg)
                            break

            # rebuild track with new times
            rebuild_track_with_messages(track, timed_messages)

        click.echo(f"time_displacement_model saving displaced file to {destination_path}")
        midi_file.save(destination_path)

        return offsets
