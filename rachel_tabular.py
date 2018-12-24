import numpy as np
import pandas as pd
import os
import torch
from fastai import *
from fastai.imports import *
from fastai.basic_train import *
from fastai.tabular import *
from fastai.metrics import *
from fastai.data_block import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing

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

        train_df.to_csv(os.path.join(
            self.data_folder, self.train_data_path), index=False)
        validate_df.to_csv(os.path.join(
            self.data_folder, self.validate_data_path), index=False)

        # Print some info about the created / loaded training data.
        print('Train shape:', train_df.shape)
        print('Train head:\n', train_df.head())
        print('Train tail:\n', train_df.tail())
        print('Train correlations:\n', train_df.corr())

        # Plot some visualisations of the training set.
        tabular_plotter.plot_data(train_df)

        return train_df, validate_df

    def load_data(self):
        print('Loading data ...')

        train_df = pd.read_csv(os.path.join(
            self.data_folder, self.train_data_path))
        validate_df = pd.read_csv(os.path.join(
            self.data_folder, self.validate_data_path))

        return train_df, validate_df

    def create_learner(self):
        print('Creating learner ...')

        # Split combined data into train and validate sets (tracking indices
        # only).
        valid_idx = range(len(self.midi_df) -
                          len(self.validate_df), len(self.midi_df))

        # Normalise output.
        self.midi_df['velocity'] = preprocessing.minmax_scale(
            np.asfarray(self.midi_df.velocity.values), feature_range=(0, 1))

        follows_pause_names = self.get_column_names_matching(
            self.midi_df, 'follows_pause')
        chord_character_names = self.get_column_names_matching(
            self.midi_df, 'chord_character')
        chord_size_names = self.get_column_names_matching(
            self.midi_df, 'chord_size')

        category_names = (['pitch_class'] + follows_pause_names +
                          chord_character_names + chord_size_names)
        continuous_names = [cat for cat in self.midi_df.columns if (
            cat not in category_names + ['velocity', 'time', 'name'])]

        dep_var = 'velocity'

        procs = [Categorify, Normalize]
        data = (TabularList.from_df(self.midi_df,
                                    path=self.data_folder,
                                    cat_names=category_names,
                                    cont_names=continuous_names,
                                    procs=procs)
                .split_by_idx(valid_idx)
                .label_from_df(cols=dep_var, label_cls=FloatList)
                .databunch())

        # For each category, use an embedding size of half of the # of possible values.
        follows_pause_szs = dict([(name, 2) for name in follows_pause_names])
        chord_character_szs = dict([(name, 6)
                                    for name in chord_character_names])
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

        learn = tabular_learner(data,
                                layers=[1000, 500],
                                emb_szs=emb_szs,
                                ps=[0.2, 0.5],
                                emb_drop=0.1,
                                y_range=y_range,
                                metrics=exp_rmspe)

        # Load the existing model if there is one.
        if os.path.isfile(os.path.join(self.data_folder, 'models', self.model_name + '.pth')):
            print('Loading model')
            learn.load(self.model_name)

        return learn

    def train(self, epochs, lr, wd):
        # learn.lr_find()
        # learn.recorder.plot()

        self.learn.fit_one_cycle(epochs, lr, wd=wd)

        # learn.show_results()
        # learn.recorder.plot_losses()

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

    def get_column_names_matching(self, df, pattern):
        '''Given a data frame and a string pattern, returns all the column names in
        the data frame containing the string.
        '''
        return [cat for cat in df.columns if pattern in cat]
