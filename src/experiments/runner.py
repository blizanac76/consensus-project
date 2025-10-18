# src/experiments/runner.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.consensus.model import ConsensusModel
from src.utils.graph_utils import spectral_gap
from src.utils.io_utils import save_experiment_results, ensure_dir

def single_run(config):
    """
    Run a single trial described by config dict:
    keys: N, graph_type, graph_params, protocol, alpha, noise_std, p_drop, seed, max_steps, tol
    Returns a dict with metrics.
    """

    start_time = time.time()
    m = ConsensusModel(
        N=config['N'],
        graph_type=config['graph_type'],
        graph_params=config.get('graph_params', {}),
        alpha=config.get('alpha', 0.5),
        protocol=config.get('protocol', 'metropolis'),
        noise_std=config.get('noise_std', 0.0),
        p_drop=config.get('p_drop', 0.0),
        seed=config.get('seed', None)
    )
    # compute spectral gap for analysis
    gap = spectral_gap(m.G)
    m.run_until(max_steps=config.get('max_steps', 1000), tol_range=config.get('tol', 1e-4))
    elapsed = time.time() - start_time
    last = m.history[-1]
    converged = m.history[-1]['range'] < config.get('tol', 1e-4)
    return {
    'N': config['N'],
    'graph_type': config['graph_type'],
    'graph_params': config.get('graph_params', {}),
    'protocol': config.get('protocol', 'metropolis'),
    'alpha': config.get('alpha', 0.5),
    'noise_std': config.get('noise_std', 0.0),
    'p_drop': config.get('p_drop', 0.0),
    'seed': config.get('seed', None),
    'spectral_gap': gap,
    'convergence_step': m.step_count,
    'final_var': last['var'],
    'final_range': last['range'],
    'final_mean': last['mean'],
    'final_l2_error': last.get('l2_error', np.nan),  # <--- add this safely
    'min_l2_error': min(h['l2_error'] for h in m.history if 'l2_error' in h),  # <--- optional, global min
    'elapsed_sec': elapsed,
    'converged': converged
}

def sweep_and_save(output_csv, param_grid, repeats=5):
    """
    param_grid: list of dicts with config template (seed will be overwritten)
    repeats: number of seeds per template
    """
    results = []
    ensure_dir(os.path.dirname(output_csv))
    for template in tqdm(param_grid, desc="Templates"):
        for r in range(repeats):
            cfg = dict(template)
            cfg['seed'] = r
            res = single_run(cfg)
            results.append(res)
    df = pd.DataFrame(results)
    save_experiment_results(df, output_csv)
    return df
