# run_experiment.py
import os
from src.experiments.runner import sweep_and_save
from src.utils.io_utils import ensure_dir
from src.viz.plotting import plot_bar_convergence
from src.viz.plots import generate_all_plots


OUT_CSV = 'data/experiment_baseline.csv'
VIS_DIR = 'visualizations'

param_grid = [
    # baseline for different topologies (metropolis)
    {'N':50, 'graph_type':'ring', 'graph_params':{}, 'protocol':'metropolis', 'alpha':0.5, 'noise_std':0.0, 'p_drop':0.2, 'max_steps':1000, 'tol':1e-3},
    {'N':50, 'graph_type':'erdos_renyi', 'graph_params':{'p':0.08}, 'protocol':'metropolis', 'alpha':0.5, 'noise_std':0.0, 'p_drop':0.0, 'max_steps':1000, 'tol':1e-3},
    {'N':50, 'graph_type':'watts_strogatz', 'graph_params':{'k':4,'p':0.1}, 'protocol':'metropolis', 'alpha':0.5, 'noise_std':0.0, 'p_drop':0.0, 'max_steps':1000, 'tol':1e-3},
    {'N':50, 'graph_type':'barabasi_albert', 'graph_params':{'m':2}, 'protocol':'metropolis', 'alpha':0.5, 'noise_std':0.0, 'p_drop':0.2, 'max_steps':1000, 'tol':1e-3},
    # compare protocols on ER
    {'N':50, 'graph_type':'erdos_renyi', 'graph_params':{'p':0.08}, 'protocol':'simple_avg', 'alpha':0.5, 'noise_std':0.0, 'p_drop':0.0, 'max_steps':1000, 'tol':1e-3},
    {'N':50, 'graph_type':'erdos_renyi', 'graph_params':{'p':0.08}, 'protocol':'gossip', 'alpha':0.5, 'noise_std':0.0, 'p_drop':0.0, 'max_steps':5000, 'tol':1e-3},
]

if __name__ == "__main__":
    ensure_dir('data')
    df = sweep_and_save(OUT_CSV, param_grid, repeats=8)
    ensure_dir(VIS_DIR)
    plot_bar_convergence(df, os.path.join(VIS_DIR, 'convergence_by_topology.png'))
    print("Done. CSV saved to:", OUT_CSV)
    generate_all_plots(OUT_CSV)
