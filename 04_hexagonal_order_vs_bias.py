import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree

import simulation_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))

SIM_PARAMS = {
    "cycle_randomness": 0.01,
    "cycle_diff": 0.5,
    "trapeffectiveheight": 0,
    "tangentialmigrationcellcycle": 0.8,
    "tangentialmigrationprob": 0.5,
    "IPratio": 0.3,
}
DISORDER_LEVELS = [0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
N_REPEATS = 100


def hexatic_order_parameter(points, k=6):
    """mean_k |psi6,k|, psi6,k = (1/6) sum_j exp(6i theta_kj) over the k nearest points j."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= k:
        return np.nan
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    psi6_local = np.zeros(n, dtype=complex)
    for i in range(n):
        vectors = points[idx[i, 1:]] - points[i]
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        psi6_local[i] = np.mean(np.exp(1j * 6 * angles))
    return float(np.mean(np.abs(psi6_local)))


def run_sweep(disorder_levels, n_repeats, sim_params, seed0=0):
    records = []
    for disorder_level in disorder_levels:
        for rep in range(n_repeats):
            seed = seed0 + int(round(disorder_level * 10000)) + rep
            rng = np.random.default_rng(seed)
            np.random.seed(seed)

            radial_fiber_list = sm.define_radial_fibers(
                sim_params["cycle_randomness"], sim_params["cycle_diff"], disorder_level, rng,
            )
            positions = np.array([rf.position for rf in radial_fiber_list])
            phase0_mask = np.array([rf.phase_group for rf in radial_fiber_list]) == 0
            psi6_phase0 = hexatic_order_parameter(positions[phase0_mask], k=6)

            cell_count, neuron_list = sm.simulation(
                radial_fiber_list,
                sim_params["trapeffectiveheight"],
                sim_params["tangentialmigrationcellcycle"],
                sim_params["tangentialmigrationprob"],
                sim_params["IPratio"],
            )
            mean0 = float(np.mean(cell_count[phase0_mask]))
            mean1 = float(np.mean(cell_count[~phase0_mask]))
            records.append({
                "disorder_level": disorder_level,
                "repeat": rep,
                "seed": seed,
                "n_neurons": len(neuron_list),
                "n_phase0": int(np.sum(phase0_mask)),
                "mean_abs_psi6_phase0": psi6_phase0,
                "mean_count_phase0": mean0,
                "mean_count_phase1": mean1,
                "cv_cell_count_all_fibers": float(np.std(cell_count) / max(np.mean(cell_count), 1e-6)),
                "column_bias": max(mean0, mean1) / max(min(mean0, mean1), 1e-3),
            })
        print(f"disorder_level={disorder_level:.4f} done ({n_repeats} repeats)", flush=True)
    return records


def plot(df, path_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    specs = [("cv_cell_count_all_fibers", "e", "CV of SCPN number\nacross radial fibers"),
             ("column_bias", "f", "Neuronal accumulation ratio\n(phase 0 vs phase pi, larger / smaller)")]
    x = "mean_abs_psi6_phase0"
    for ax, (y, letter, ylabel) in zip(axes, specs):
        sc = ax.scatter(df[x], df[y], c=df["disorder_level"], cmap="viridis", s=20, alpha=0.5)
        centers = df.groupby("disorder_level")[[x, y]].mean()
        ax.plot(centers[x], centers[y], "-o", color="magenta", lw=2, ms=4, zorder=5)
        r, p = stats.pearsonr(df[x], df[y])
        ax.set_title(f"r = {r:.2f}, p = {p:.1e}")
        ax.set_xlabel("Hexagonal order of phase-synchronized RGCs\n(mean |psi6|)")
        ax.set_ylabel(ylabel)
        ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
        fig.colorbar(sc, ax=ax, label="Disorder level")
        print(f"{y}: Pearson r = {r:.3f}, p = {p:.2e}")
    fig.savefig(path_stem + ".png", dpi=200)
    fig.savefig(path_stem + ".eps")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=0, help="base random seed (default 0 = published run)")
    parser.add_argument("--random-seed", action="store_true", help="use a random base seed")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.DataFrame(run_sweep(DISORDER_LEVELS, args.repeats, SIM_PARAMS, sm.resolve_seed(None if args.random_seed else args.seed)))
    df.to_csv(os.path.join(args.out_dir, "ed_fig6e-f_hexagonal_order.csv"), index=False)
    plot(df, os.path.join(args.out_dir, "ed_fig6e-f_hexagonal_order"))
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
