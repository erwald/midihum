import numpy as np
import pandas as pd
import os
import torch
import re
from fastai import *
from fastai.imports import *
from fastai.basic_train import *
from fastai.tabular import *
from fastai.metrics import *
from fastai.data_block import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from mido import MidiFile

from directories import *
from midi_dataframe_converter import midi_files_to_data_frame
import tabular_plotter


class RachelTabular:
    '''A tabular neural network for predicting velocities of MIDI files.
    '''

    def __init__(self, prepare_data):
        '''Initialisation.'''
        self.train_data_path = 'train_data.csv'
        self.validate_data_path = 'validate_data.csv'
        self.data_folder = './data'
        self.model_name = 'tabular_model'

        # Load data.
        if prepare_data:
            self.train_df, self.validate_df = self.prepare_data()
        else:
            self.train_df, self.validate_df = self.load_data()
        self.midi_df = pd.concat([self.train_df, self.validate_df])

        self.learn = self.create_learner()

    def prepare_data(self):
        print('Preparing data ...')

        midi_data_filepaths = get_files(midi_data_valid_path, ['.mid', '.MID'])

        train_filepaths, validate_filepaths = train_test_split(
            midi_data_filepaths, test_size=0.1, random_state=1988)

        train_df = midi_files_to_data_frame(midi_filepaths=train_filepaths)
        validate_df = midi_files_to_data_frame(
            midi_filepaths=validate_filepaths)

        processed_file_count = len(train_df) + len(validate_df)
        print(f'Processed {processed_file_count} files; now saving ...')

        train_df.to_csv(os.path.join(self.data_folder,
                                     self.train_data_path), index=False, encoding='utf-8')
        validate_df.to_csv(os.path.join(
            self.data_folder, self.validate_data_path), index=False, encoding='utf-8')

        # Print some info about the created / loaded training data.
        print('Train shape:', train_df.shape)
        print('Train head:\n', train_df.head())
        print('Train tail:\n', train_df.tail())
        print('Train velocity correlations:\n',
              train_df.corr().velocity.sort_values(ascending=False))

        # Plot some visualisations of the training set.
        tabular_plotter.plot_data(train_df)

        return train_df, validate_df

    def load_data(self):
        print('Loading data ...')

        train_df = pd.read_csv(os.path.join(
            self.data_folder, self.train_data_path), encoding='utf-8')
        validate_df = pd.read_csv(os.path.join(
            self.data_folder, self.validate_data_path), encoding='utf-8')

        return train_df, validate_df

    def create_learner(self):
        print('Creating learner ...')

        # Split combined data into train and validate sets (tracking indices
        # only).
        valid_idx = range(len(self.midi_df) -
                          len(self.validate_df), len(self.midi_df))

        # Scale output.
        self.midi_df['velocity'] = preprocessing.minmax_scale(
            np.asfarray(self.midi_df.velocity.values), feature_range=(-1, 1))

        # Define names of categorical columns (including lags and excluding non-
        # categorical columns such as "time elapsed since X").
        follows_pause_names = self.get_column_names_matching(
            self.midi_df, 'follows_pause(_pressed|\_(lag|fwd_lag)\_\d)?')
        chord_character_names = self.get_column_names_matching(
            self.midi_df, '^chord_character(?!_occur)(_pressed|\_(lag|fwd_lag)\_\d)?')
        chord_size_names = self.get_column_names_matching(
            self.midi_df, '^chord_size(?!_occur)(_pressed|\_(lag|fwd_lag)\_\d)?')
        category_names = (['pitch_class'] + follows_pause_names +
                          chord_character_names + chord_size_names)

        # Define names of continuous columns.
        columns_to_skip = ['velocity', 'time',
                           'midi_track_index', 'midi_event_index', 'name']
        continuous_names = [cat for cat in self.midi_df.columns if (
            cat not in category_names + columns_to_skip)]

        dep_var = 'velocity'

        data = (TabularList.from_df(self.midi_df,
                                    path=self.data_folder,
                                    cat_names=category_names,
                                    cont_names=continuous_names,
                                    procs=[FillMissing, Categorify, Normalize])
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList)
                .databunch())

        # Create a range between which all of our output values should be. (We
        # set the upper bound to 1.2 because of the last layer being a sigmoid,
        # meaning it is very unlikely to reach the extremes.)
        y_range = torch.tensor([-1.2, 1.2], device=defaults.device)

        learn = tabular_learner(data,
                                layers=[1000, 500],
                                ps=[1e-2, 1e-1],
                                emb_drop=0.04,
                                y_range=y_range,
                                metrics=exp_rmspe)

        # Load the existing model if there is one.
        model_path = os.path.join(
            self.data_folder, 'models', self.model_name + '.pth')
        if os.path.isfile(model_path):
            print(f'Loading saved model from {model_path} ...')
            learn.load(self.model_name)

        return learn

    def train(self, epochs, lr, wd):
        self.learn.fit_one_cycle(epochs, lr, wd=wd)

        # Save the model.
        print('Saving model ...')
        self.learn.save(self.model_name)

        self.predict_validation_data()

    def predict_validation_data(self):
        predictions, targets = [x.numpy().flatten()
                                for x in self.learn.get_preds(DatasetType.Valid)]
        prediction_df = pd.DataFrame(
            {'name': self.validate_df.name, 'prediction': predictions, 'target': targets})
        prediction_df['error'] = (prediction_df.target -
                                  prediction_df.prediction).abs()
        print('Prediction range:', (np.amin(predictions), np.amax(predictions)))
        print('Predictions:', prediction_df.head())

        tabular_plotter.plot_predictions(prediction_df)

    def humanize(self, midi_filepath, quantization=4):
        df = midi_files_to_data_frame(midi_filepaths=[midi_filepath])

        # Print some information about the input data.
        print('Input shape:', df.shape)
        print('Input head:\n', df.head())
        print('Input tail:\n', df.tail())

        # Make velocity predictions for each row (note on) of the input.
        df['prediction'] = [self.learn.predict(row)[2].numpy().flatten()[0]
                            for _, row in df.iterrows()]

        # Load input MIDI file and, for each prediction, set the new velocity.
        midi_file = MidiFile(midi_filepath)
        for _, row in df.iterrows():
            velocity = max(min(round((row.prediction + 1) / 2 * 127), 127), 1)
            midi_file.tracks[row.midi_track_index][row.midi_event_index].velocity = velocity

        # Save the MIDI file with the new velocities to the output directory.
        out_path = os.path.join(output_dir, os.path.split(midi_filepath)[-1])
        print(f'Saving humanized file to {out_path}')
        midi_file.save(out_path)

    def get_column_names_matching(self, df, pattern):
        '''Given a data frame and a string regex pattern, returns all the column
        names in the data frame containing the string.
        '''
        return [col for col in df.columns if re.match(pattern, col)]
