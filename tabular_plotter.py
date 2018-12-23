import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from directories import *

sns.set()


def plot_data(df):
    '''Produces various plots of the given data frame.
    '''
    print('Plotting data ...')

    cat_names = ['pitch_class',
                 'octave',
                 'follows_pause',
                 'chord_character_pressed',
                 'chord_size_pressed',
                 'chord_character',
                 'chord_size']
    cont_names = ['velocity',
                  'pitch',
                  'interval_from_pressed',
                  'interval_from_released',
                  'sustain',
                  'time_since_last_pressed',
                  'time_since_last_released',
                  'time_since_pitch_class',
                  'time_since_octave',
                  'time_since_pause',
                  'time_since_chord_character',
                  'time_since_chord_size']

    # Box-and-whispers plots of categorical variables.
    for col in cat_names:
        plot = sns.boxplot(x=col, y='velocity', data=df)
        plot.get_figure().savefig(os.path.join(
            model_output_dir, 'boxplot_{}.png'.format(col)))
        plt.clf()

    # Count plots.
    for col in cat_names:
        plot = sns.countplot(x=col, palette='rocket', data=df)
        plot.get_figure().savefig(os.path.join(
            model_output_dir, 'countplot_{}.png'.format(col)))
        plt.clf()

    # Distribution plots.
    for col in cont_names:
        plot = sns.distplot(df[col])
        plot.get_figure().savefig(os.path.join(
            model_output_dir, 'distplot_{}.png'.format(col)))
        plt.clf()

    # Bar plots for categorical values against velocity.
    for col in cat_names:
        plot = sns.barplot(x=df[col], y=df.velocity, palette='rocket')
        plot.get_figure().savefig(os.path.join(
            model_output_dir, 'barplot_{}_vs_velocity.png'.format(col)))
        plt.clf()

    # Hex + dist plots for continuous names against velocity.
    for col in [name for name in cont_names if name != 'velocity']:
        plot = sns.jointplot(x=col, y='velocity', data=df, kind='hex')
        plot.savefig(os.path.join(model_output_dir,
                                  'hexplot_{}_vs_velocity.png'.format(col)))
        plt.clf()

    plt.subplots(figsize=(30, 30))

    # Heatmap of correlations.
    plot = sns.heatmap(df.corr(), vmin=-1, vmax=1,
                       cmap='PiYG', xticklabels=True, yticklabels=True)
    plt.tight_layout()
    plot.get_figure().savefig(os.path.join(
        model_output_dir, 'regression_correlations.png'))
    plt.clf()

    # Heatmap of absolute correlations (iow, ignoring whether the correlation
    # is negative or positive and looking only at its strength).
    plot = sns.heatmap(np.abs(df.corr()), vmin=0, vmax=1,
                       cmap='Blues', xticklabels=True, yticklabels=True)
    plt.tight_layout()
    plot.get_figure().savefig(os.path.join(
        model_output_dir, 'regression_correlations_absolute.png'))
    plt.clf()


def plot_predictions(df):
    '''Plot relationship between predictions and targets.
    '''
    print('Plotting predictions ...')

    # Relationship between predictions and targets.
    plot = sns.relplot(x='target', y='prediction', col='name', hue='error',
                       col_wrap=5, data=df)
    plot.set(xlim=(0, 1), ylim=(0, 1))
    plot.savefig(os.path.join(model_output_dir, 'predictions.png'))
