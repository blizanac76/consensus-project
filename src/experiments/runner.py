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
    pokrece jednu simulaciju sa datim parametrima (config je dict)
    vraca metrike iz simulacije kao dict
    """

    start_time = time.time()  # vreme pocetka pokretanja
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
     # racuna se spektralni zazor/gap grafa (lambda_2)
    gap = spectral_gap(m.G)
    # pokrece se simulacija dok ne konvergira ili dok ne stigne do max_steps
    m.run_until(max_steps=config.get('max_steps', 1000), tol_range=config.get('tol', 1e-4))
    elapsed = time.time() - start_time
     # uzima se poslednji zapis 
    last = m.history[-1]
     # da li je model konvergirao
    converged = m.history[-1]['range'] < config.get('tol', 1e-4)

    # formira se dict sa metrikama
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
    pokrece vise simulacija razliciti grafovi i protokoli
    param_grid: lista dict-ova sa kombinacijama parametara
    repeats: koliko puta se ponavlja svaka konfiguracija (vise random seed)
    """
    results = []  # lista svih rezultata
    ensure_dir(os.path.dirname(output_csv))  # pravi folder ako ne postoji za ispis metrike u csv
    # iterira kroz sve konfiguracije (param_grid), prikazuje progres bar
    for template in tqdm(param_grid, desc="Templates"):
        for r in range(repeats):  
            cfg = dict(template)  # pravi kopiju konfiguracije
            cfg['seed'] = r  # postavlja random seed
            res = single_run(cfg)  
            results.append(res)  # dodaje rezultat
    df = pd.DataFrame(results)  # pretvara rezultate u pandas tabelu
    save_experiment_results(df, output_csv)  # cuva kao CSV
    return df  # vraca tabelu rezultata
