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
from sklearn import metrics
from sklearn.model_selection import train_test_split

from midi_dataframe_converter import midi_files_to_data_frame
from directories import *


quantization = 4

# Load data.
midi_data_filepaths = get_files(
    midi_data_valid_quantized_path, ['.mid', '.MID'])
train_filepaths, validate_filepaths = train_test_split(
    midi_data_filepaths, test_size=0.1, random_state=1988)

train_df = midi_files_to_data_frame(midi_filepaths=train_filepaths,
                                    quantization=quantization)
validate_df = midi_files_to_data_frame(midi_filepaths=validate_filepaths,
                                       quantization=quantization)
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

category_names = ['pitch_class']
continuous_names = ['pitch', 'octave', 'velocity_2']
dep_var = 'velocity'
y_range = range(-1, 1)

procs = [Categorify, Normalize]
data = (TabularList.from_df(midi_df, path=data_folder, cat_names=category_names, cont_names=continuous_names, procs=procs)
        .split_by_idx(valid_idx)
        .label_from_df(cols=dep_var, label_cls=FloatList)
        .databunch())

emb_szs = {'pitch_class': 12}
learn = tabular_learner(data, layers=[200, 100], emb_szs=emb_szs, ps=[
    0.001, 0.01], emb_drop=0.04, y_range=y_range, metrics=exp_rmspe)

# learn.lr_find()
# learn.recorder.plot()

learn.fit_one_cycle(5, 1e-4)
# learn.show_results()

predictions, targets = learn.get_preds(DatasetType.Valid)
print('Predictions: ', predictions)
print('Expected: ', targets)

sns.set()
plot = sns.heatmap(train_df.corr())
plot.get_figure().savefig(os.path.join(
    model_output_dir, 'regression_correlations.png'))
