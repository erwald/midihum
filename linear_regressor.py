import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
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


# Load data.
midi_data_filepaths = get_files(
    midi_data_valid_quantized_path, ['.mid', '.MID'])
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
    midi_df.velocity.values, feature_range=(-1, 1))

category_names = ['pitch_class',
                  'follows_pause']
continuous_names = ['pitch',
                    'octave',
                    'nearness_to_end',
                    'nearness_to_midpoint',
                    'number_of_notes',
                    'num_of_notes_adj_by_dur',
                    'interval_from_released',
                    'interval_from_pressed',
                    'sustain',
                    'song_duration',
                    'time_since_last_note',
                    'mean_sustain',
                    'sustain_adj_by_mean']
dep_var = 'velocity'

procs = [Categorify, Normalize]
data = (TabularList.from_df(midi_df, path=data_folder, cat_names=category_names, cont_names=continuous_names, procs=procs)
        .split_by_idx(valid_idx)
        .label_from_df(cols=dep_var, label_cls=FloatList)
        .databunch())

# For each category, use an embedding size of half of the # of possible values.
category_szs = {'pitch_class': 12, 'follows_pause': 2}
emb_szs = {k: v // 2 for k, v in category_szs.items()}

learn = tabular_learner(
    data, layers=[200, 100], emb_szs=emb_szs, y_range=None, metrics=exp_rmspe)

# learn.lr_find()
# learn.recorder.plot()

learn.fit_one_cycle(5, 1e-4)
# learn.show_results()

predictions, targets = [x.numpy().flatten()
                        for x in learn.get_preds(DatasetType.Valid)]
prediction_df = pd.DataFrame(
    {'name': validate_df.name, 'prediction': predictions, 'target': targets})
print('Prediction range:', (np.amin(predictions), np.amax(predictions)))
print('Predictions:', prediction_df.head())

print('Generating plots ...')

sns.set()

plot = sns.heatmap(train_df.corr(), vmin=-1, vmax=1, cmap="PiYG")
plt.tight_layout()
plot.get_figure().savefig(os.path.join(
    model_output_dir, 'regression_correlations.png'))
plt.clf()

plot = sns.heatmap(np.abs(train_df.corr()), vmin=0, vmax=1, cmap="Blues")
plt.tight_layout()
plot.get_figure().savefig(os.path.join(
    model_output_dir, 'regression_correlations_absolute.png'))
plt.clf()

plot = sns.pairplot(train_df)
plot.savefig(os.path.join(model_output_dir, 'pairwise_relationships.png'))
plt.clf()

# Plot relationship between predictions and targets.
plot = sns.relplot(x='target', y='prediction', col='name',
                   col_wrap=5, data=prediction_df)
plot.set(xlim=(-1, 1), ylim=(-1, 1))
plot.savefig(os.path.join(model_output_dir, 'predictions.png'))
