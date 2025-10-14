# src/viz/plotting.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_time_series(history_df, out_path):
    """Plot mean & range from history dataframe (list of dicts)."""
    ensure_dir(os.path.dirname(out_path))
    df = pd.DataFrame(history_df)
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df['step'], df['mean'], label='mean')
    ax1.set_xlabel('step'); ax1.set_ylabel('mean')
    ax2 = ax1.twinx()
    ax2.plot(df['step'], df['range'], label='range', linestyle='--')
    ax2.set_ylabel('range')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.title('Consensus: mean (left) and range (right)')
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_bar_convergence(df, out_path, groupby='graph_type'):
    """
    df: result dataframe from runner.sweep_and_save
    produce bar plot: mean convergence_step grouped by groupby
    """
    ensure_dir(os.path.dirname(out_path))
    stats = df.groupby(groupby)['convergence_step'].agg(['mean','std','count']).reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(stats[groupby], stats['mean'], yerr=stats['std'], capsize=5)
    ax.set_ylabel('steps to consensus (mean ± std)')
    ax.set_xlabel(groupby)
    plt.title('Convergence time by {}'.format(groupby))
    plt.savefig(out_path)
    plt.close(fig)

# helper to ensure dir
def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
