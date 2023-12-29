from pathlib import Path
from typing import List, Optional, Tuple

import click
import mido
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import sklearn

from midi_to_df_conversion import midi_files_to_df
from prepare_midi import load_data


class MidihumModel:
    """An XGBoost model for predicting velocities of MIDI note values."""

    model_cache_path = Path("model_cache")
    model_path = Path(f"{model_cache_path}/midihum.json")
    scaler_path = Path(f"{model_cache_path}/midihum_scaler.json")

    def __init__(self):
        if self.model_path.exists() and self.scaler_path.exists():
            click.echo(
                f"midihum_model loading model from {self.model_path} and {self.scaler_path}"
            )
            self.model = xgb.XGBRegressor(
                booster="gbtree",
                max_depth=7,
                learning_rate=0.05,
                n_estimators=1700,
                gamma=0.1,
                min_child_weight=7,
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
            click.echo(
                f"midihum_model could not find model in {self.model_path} and {self.scaler_path}"
            )
            raise Exception()

    @staticmethod
    def _get_column_names_from_df(
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str], List[str]]:
        columns_to_skip = [
            "velocity",
            "time",
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
        out_names = ["velocity"]
        return (category_names, continuous_names, out_names)

    @staticmethod
    def _rescale_predictions(
        scaler: sklearn.preprocessing.StandardScaler, preds: pd.Series
    ) -> pd.Series:
        # FIXME: get velocity idx dynamically -- maybe save it with the scaler.
        velocity_idx = 0
        # HACK: not sure why 2, it just seems to work well ...
        return np.clip(
            preds * 2 * np.sqrt(scaler.var_[velocity_idx]) + scaler.mean_[velocity_idx],
            1,
            127,
        )

    def add_velocities_to_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        cat_names, cont_names, out_names = self._get_column_names_from_df(df)
        for col in cat_names:
            df[col] = df[col].astype("category")

        # standardize input columns
        out_cols = df[out_names].copy()
        df[cont_names + out_names] = self.scaler.transform(df[cont_names + out_names])

        # make velocity predictions for each row (note on) of the input
        df = df.drop(out_names, axis=1)
        df["prediction"] = self.model.predict(
            df.drop(["midi_track_index", "midi_event_index", "name"], axis=1)
        )
        df["prediction"] = self._rescale_predictions(self.scaler, df["prediction"])
        df[out_names] = out_cols
        click.echo(
            f"midihum_tabular inferred {len(df)} velocities with mean {np.mean(df.prediction)} and std {np.std(df.prediction)}"
        )
        return df

    def humanize(
        self, source_path: Path, destination_path: Path, rescale: bool = True
    ) -> List[float]:
        click.echo(f"midihum_tabular humanizing {source_path}")
        df = midi_files_to_df(
            midi_filepaths=[source_path], skip_suspicious=False
        ).copy()

        # standardize input columns and add predicted velocities
        df = self.add_velocities_to_df(df)

        # load input midi file and, for each prediction, set the new velocity
        midi_file = mido.MidiFile(source_path)
        velocities = [round(row.prediction) for _, row in df.iterrows()]
        for row, velocity in zip(df.itertuples(), velocities):
            midi_file.tracks[row.midi_track_index][
                row.midi_event_index
            ].velocity = velocity

        click.echo(f"midihum_tabular saving humanized file to {destination_path}")
        midi_file.save(destination_path)

        return velocities
