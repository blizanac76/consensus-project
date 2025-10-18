# src/visualizations/plots.py ne radi
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils.io_utils import ensure_dir


def plot_convergence_by_topology(df, out_dir):
    """
    Compare average convergence step vs. topology (per protocol).
    """
    ensure_dir(out_dir)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="graph_type",
        y="convergence_step",
        hue="protocol",
        ci="sd",
        palette="viridis"
    )
    plt.ylabel("Average convergence steps")
    plt.xlabel("Graph topology")
    plt.title("Consensus Convergence Speed by Topology and Protocol")
    plt.legend(title="Protocol")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "convergence_by_topology.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def plot_error_decay(csv_path, out_dir, graph_type="erdos_renyi", protocol="metropolis"):
    """
    Plot L2 error decay over steps for one example run (shows convergence dynamics).
    """
    from src.consensus.model import ConsensusModel

    ensure_dir(out_dir)
    # Load config from CSV row
    df = pd.read_csv(csv_path)
    row = df[(df["graph_type"] == graph_type) & (df["protocol"] == protocol)].iloc[0]

    # Re-run one model to collect detailed history
    m = ConsensusModel(
        N=row.N,
        graph_type=row.graph_type,
        graph_params=eval(str(row.graph_params)),
        alpha=row.alpha,
        protocol=row.protocol,
        noise_std=row.noise_std,
        p_drop=row.p_drop,
        seed=int(row.seed) if not np.isnan(row.seed) else None,
    )
    m.run_until(max_steps=1000)

    hist = pd.DataFrame(m.history)
    plt.figure(figsize=(7, 5))
    plt.semilogy(hist["step"], hist["l2_error"], label=f"{graph_type}-{protocol}")
    plt.xlabel("Simulation Step")
    plt.ylabel("L2 Consensus Error (log scale)")
    plt.title("Error Decay Over Time")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"error_decay_{graph_type}_{protocol}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def plot_runtime_vs_topology(df, out_dir):
    """
    Compare runtime across graph types and protocols.
    """
    ensure_dir(out_dir)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="graph_type",
        y="elapsed_sec",
        hue="protocol",
        ci="sd",
        palette="mako"
    )
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Graph topology")
    plt.title("Computation Time by Topology and Protocol")
    plt.legend(title="Protocol")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "runtime_by_topology.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def plot_correlation(df, out_dir):
    """
    Scatter plot: spectral gap vs convergence steps.
    Demonstrates relation between connectivity and convergence speed.
    """
    ensure_dir(out_dir)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="spectral_gap",
        y="convergence_step",
        hue="graph_type",
        style="protocol",
        s=70
    )
    plt.xlabel("Spectral Gap (λ2)")
    plt.ylabel("Convergence Steps")
    plt.title("Spectral Gap vs Convergence Speed")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "gap_vs_convergence.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def generate_all_plots(csv_path, out_dir="visualizations/summary"):
    """
    Generate all key plots from results CSV.
    """
    df = pd.read_csv(csv_path)
    ensure_dir(out_dir)

    # Drop weird rows with missing values
    df = df.dropna(subset=["spectral_gap", "convergence_step"])

    plot_convergence_by_topology(df, out_dir)
    plot_runtime_vs_topology(df, out_dir)
    plot_correlation(df, out_dir)

    # Create detailed L2 error decay example
    plot_error_decay(csv_path, out_dir, graph_type="erdos_renyi", protocol="metropolis")

    print("\n✅ All plots generated in:", out_dir)
