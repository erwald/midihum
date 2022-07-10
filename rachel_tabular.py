import os
from pathlib import Path
from typing import List, Optional, Tuple

import click
from fastai.metrics import rmse, exp_rmspe
from fastai.tabular.core import Categorify, FillMissing, Normalize
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import tabular_learner, load_learner
from mido import MidiFile
from sklearn import preprocessing
import numpy as np
import pandas as pd

from midi_to_df_conversion import midi_files_to_df
from prepare_midi import load_data
import tabular_plotter


class RachelTabular:
    """A tabular neural network for predicting velocities of MIDI files.
    """
    model_dir = Path("model_cache")

    def __init__(
            self, name: str = "rachel", data_dir: Path = Path("dfs"), layers: Optional[List[int]] = None,
            predict_only: bool = False):
        self.model_filename = Path(f"{name}.pickle")
        if not predict_only:
            self.train_df, self.validate_df = load_data(data_dir)
        else:
            self.train_df = None
            self.validate_df = None

        os.makedirs(self.model_dir, exist_ok=True)
        self._create_learner(layers=layers, include_data=(not predict_only))

    def _save_model(self):
        click.echo(f"rachel_tabular saving model to {self.model_filename}")
        self.learn.export(self.model_filename)

    def _load_model_if_exists(self) -> bool:
        path = self.model_dir / self.model_filename
        if path.exists():
            click.echo(f"rachel_tabular loading model from {path}")
            self.learn = load_learner(path)
            return True
        click.echo(f"rachel_tabular couldn't find model at {path}")
        return False

    @staticmethod
    def _get_column_names_from_df(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        columns_to_skip = ["velocity", "time", "midi_track_index", "midi_event_index", "name"]
        category_names = [
            col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col not in columns_to_skip]
        continuous_names = [col for col in df.columns if col not in category_names + columns_to_skip]
        return (category_names, continuous_names)

    def _create_data_loaders(self) -> TabularDataLoaders:
        click.echo("rachel_tabular creating data loaders")

        self.train_df.velocity = \
            preprocessing.minmax_scale(np.asfarray(self.train_df.velocity.values), feature_range=(-1, 1))
        self.validate_df.velocity = \
            preprocessing.minmax_scale(np.asfarray(self.validate_df.velocity.values), feature_range=(-1, 1))

        df = pd.concat([self.train_df, self.validate_df])
        category_names, continuous_names = self._get_column_names_from_df(self.train_df)
        return TabularDataLoaders.from_df(
            df=df, path=str(self.model_dir), procs=[Categorify, FillMissing, Normalize], cat_names=category_names,
            cont_names=continuous_names, y_names="velocity", valid_idx=list(range(len(self.train_df), len(df))), bs=64)

    def _create_learner(self, layers: Optional[List[int]], include_data: bool):
        click.echo(f"rachel_tabular creating learner with layers={layers} and include_data={include_data}")
        if not include_data and self._load_model_if_exists():
            return

        assert include_data, "couldn't find cached model"
        assert layers, layers
        dls = self._create_data_loaders()
        # set y_range to slightly more than (-1, 1) because the last layer is a sigmoid, meaning it's unlikely to reach
        # the extremes.
        self.learn = tabular_learner(dls=dls, layers=layers, y_range=(-1.2, 1.2), metrics=[rmse, exp_rmspe])

    def train(
            self, epochs: int = 3, lr: float = 0.001, wd: float = 0.7, plot_dir: Optional[Path] = None,
            save_model: bool = True):
        click.echo(f"rachel_tabular training for {epochs} epochs with learning rate {lr} and weight decay {wd}")
        self.learn.fit(epochs, lr=lr, wd=wd)
        if save_model:
            self._save_model()
        if plot_dir:
            self.predict_validation_data(plot_dir)

    @staticmethod
    def _rescale_predictions(preds: pd.Series) -> pd.Series:
        return (preds - preds.mean()) / (preds.std() * 2.5) # not sure why 2.5, it just seems to work well ...

    def predict_validation_data(self, plot_dir: Path) -> pd.DataFrame:
        assert self.train_df is not None, self.train_df
        assert self.validate_df is not None, self.validate_df
        self.learn.dls.test_dl(self.validate_df)
        predictions, targets = [x.numpy().flatten() for x in self.learn.get_preds()]
        prediction_df = pd.DataFrame({"name": self.validate_df.name, "prediction": predictions, "target": targets})
        prediction_df = prediction_df.reset_index(drop=True)
        prediction_df["adjusted_prediction"] = \
            prediction_df.groupby("name") \
                         .apply(lambda g: self._rescale_predictions(g.prediction)) \
                         .reset_index("name", drop=True)

        prediction_df["error"] = (prediction_df.target - prediction_df.prediction).abs()
        prediction_df["adjusted_error"] = (prediction_df.target - prediction_df.adjusted_prediction).abs()

        click.echo(f"prediction range: {(np.amin(predictions), np.amax(predictions))}")
        click.echo(f"predictions:\n{prediction_df.head()}")
        click.echo("prediction-target correlations:")
        for name in prediction_df.name.unique():
            song_df = prediction_df[prediction_df.name == name]
            correlation = song_df.prediction.corr(song_df.target)
            adjusted_correlation = song_df.adjusted_prediction.corr(song_df.target)
            click.echo(f"{name}: {correlation} (normal), {adjusted_correlation} (adjusted)")
        total_adjusted_correlation = prediction_df.adjusted_prediction.corr(prediction_df.target)
        click.echo(f"total (adjusted) prediction-target correlation: {total_adjusted_correlation}")
        tabular_plotter.plot_predictions(prediction_df, plot_dir)

        return prediction_df

    def humanize(self, source_path: Path, destination_path: Path, rescale: bool = True) -> List[float]:
        click.echo(f"rachel_tabular humanizing {source_path}")
        df = midi_files_to_df(midi_filepaths=[source_path], skip_suspicious=False)
        df = df.drop("velocity", axis=1)
        click.echo(f"input shape: {df.shape}")

        # make velocity predictions for each row (note on) of the input
        dl = self.learn.dls.test_dl(df)
        df["prediction"] = self.learn.get_preds(dl=dl)[0].numpy().flatten()
        if rescale:
            df.prediction = self._rescale_predictions(df.prediction)

        min_velocity = df.prediction.min()
        max_velocity = df.prediction.max()
        click.echo(f"rachel_tabular got {df.count()[0]} velocities in range ({min_velocity} ... {max_velocity})")

        # load input midi file and, for each prediction, set the new velocity
        midi_file = MidiFile(source_path)
        velocities = [max(1, min(127, round(((row.prediction + 1.0) / 2.0) * 127.0))) for _, row in df.iterrows()]
        for row, velocity in zip(df.itertuples(), velocities):
            midi_file.tracks[row.midi_track_index][row.midi_event_index].velocity = velocity

        click.echo(f"rachel_tabular saving humanized file to {destination_path}")
        midi_file.save(destination_path)

        return velocities
