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

    plt.subplots(figsize=(20, 20))

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
