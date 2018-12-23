import numpy as np
import pandas as pd
import os
import glob
import torch
from fastai import *
from fastai.imports import *
from fastai.basic_train import *
from fastai.tabular import *
from fastai.metrics import *
from fastai.data_block import *
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

from midi_dataframe_converter import midi_files_to_data_frame
from directories import *
import tabular_plotter


# Load data.
midi_data_filepaths = get_files(midi_data_valid_path, ['.mid', '.MID'])
train_filepaths, validate_filepaths = train_test_split(
    midi_data_filepaths, test_size=0.1, random_state=1988)

train_df = midi_files_to_data_frame(midi_filepaths=train_filepaths)
validate_df = midi_files_to_data_frame(midi_filepaths=validate_filepaths)
midi_df = pd.concat([train_df, validate_df])

data_folder = './data'
train_df.to_csv(os.path.join(data_folder, 'train_data.csv'), index=False)
validate_df.to_csv(os.path.join(data_folder, 'validate_data.csv'), index=False)

print('Train shape:', train_df.shape)
print('Train head:\n', train_df.head())
print('Train tail:\n', train_df.tail())
print('Train correlations:\n', train_df.corr())

# Split data into train and validate sets.
valid_idx = range(len(midi_df) - len(validate_df), len(midi_df))

# Normalise output.
midi_df['velocity'] = preprocessing.minmax_scale(
    midi_df.velocity.values, feature_range=(0, 1))


def get_column_names_matching(df, str):
    '''Given a data frame and a string pattern, returns all the column names in
    the data frame containing the string.
    '''
    return [cat for cat in df.columns if str in cat]


follows_pause_names = get_column_names_matching(midi_df, 'follows_pause')
chord_character_names = get_column_names_matching(midi_df, 'chord_character')
chord_size_names = get_column_names_matching(midi_df, 'chord_size')

category_names = (['pitch_class'] + follows_pause_names +
                  chord_character_names + chord_size_names)
continuous_names = [cat for cat in midi_df.columns if (
    cat not in category_names + ['velocity', 'time', 'name'])]

dep_var = 'velocity'

procs = [Categorify, Normalize]
data = (TabularList.from_df(midi_df, path=data_folder, cat_names=category_names, cont_names=continuous_names, procs=procs)
        .split_by_idx(valid_idx)
        .label_from_df(cols=dep_var, label_cls=FloatList)
        .databunch())

# For each category, use an embedding size of half of the # of possible values.
follows_pause_szs = dict([(name, 2) for name in follows_pause_names])
chord_character_szs = dict([(name, 6) for name in chord_character_names])
chord_size_szs = dict([(name, 7) for name in chord_size_names])
category_szs = {'pitch_class': 12,
                'follows_pause': 2,
                **follows_pause_szs,
                **chord_character_szs,
                **chord_size_szs}
emb_szs = {k: (v + 1) // 2 for k, v in category_szs.items()}

# Create a range between which all of our output values should be. (We set the
# upper bound to 1.2 because of the last layer being a sigmoid, meaning it is
# very unlikely to reach the extremes.)
y_range = torch.tensor([0, 1.2], device=defaults.device)

learn = tabular_learner(data, layers=[1000, 500], emb_szs=emb_szs, ps=[
                        2e-1, 5e-1], emb_drop=1e-1, y_range=y_range, metrics=exp_rmspe)

# learn.lr_find()
# learn.recorder.plot()

learn.fit_one_cycle(3, 1e-3)

# learn.show_results()
# learn.recorder.plot_losses(last=-1)

predictions, targets = [x.numpy().flatten()
                        for x in learn.get_preds(DatasetType.Valid)]
prediction_df = pd.DataFrame(
    {'name': validate_df.name, 'prediction': predictions, 'target': targets})
prediction_df['error'] = (prediction_df.target -
                          prediction_df.prediction).abs()
print('Prediction range:', (np.amin(predictions), np.amax(predictions)))
print('Predictions:', prediction_df.head())

tabular_plotter.plot_data(train_df)
tabular_plotter.plot_predictions(prediction_df)
