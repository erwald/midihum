import os
from pathlib import Path

import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="Set3")

def plot_data(df: pd.DataFrame, output_dir: Path):
    click.echo("tabular_plotter plotting data")
    os.makedirs(output_dir, exist_ok=True)

    categorical_cols = []
    continuous_cols = []
    for col in df.columns:
        blacklist = ["_lag", "_sma", "_ewm", "_bollinger", "chikou", "tenkan", "kijun", "senkou", "cloud"]
        if col in ["name", "midi_event_index", "midi_track_index"] or any(s in col for s in blacklist):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in categorical_cols:
        plot = sns.boxplot(x=col, y="velocity", data=df)
        plot.get_figure().savefig(output_dir / f"boxplot_{col}.png")
        plt.clf()

    for col in categorical_cols:
        plot = sns.countplot(x=col, data=df)
        plot.get_figure().savefig(output_dir / f"countplot_{col}.png")
        plt.clf()

    for col in continuous_cols:
        plot = sns.regplot(x=col, y="velocity", marker="+", scatter_kws={"alpha": 0.25}, data=df)
        plot.get_figure().savefig(output_dir / f"regplot_{col}.png")
        plt.clf()

    for col in continuous_cols:
        plot = sns.histplot(df[col])
        plot.get_figure().savefig(output_dir / f"histplot_{col}.png")
        plt.clf()

def plot_predictions(df: pd.DataFrame, output_dir: Path):
    click.echo("tabular_plotter plotting predictions")
    os.makedirs(output_dir, exist_ok=True)

    # relationship between predictions and targets
    for col, err_col in [("prediction", "error"), ("adjusted_prediction", "adjusted_error")]:
        g = sns.FacetGrid(df, col="name", col_wrap=8)
        g.map_dataframe(sns.scatterplot, x="target", y=col, hue=err_col, legend=False)
        g.set(xlim=(-1, 1), ylim=(-1, 1))
        g.savefig(output_dir / f"{col}.png")
