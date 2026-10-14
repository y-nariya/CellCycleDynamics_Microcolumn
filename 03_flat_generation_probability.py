import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import simulation_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))
N_TIME_BINS = sm.N_TIME_BINS
DURATION = sm.GENERATION_DURATION_BINS
N_TYPE0 = sm.N_TYPE0


def build_flat_prob(prob, duration_bins):
    return np.concatenate([np.full(duration_bins, prob), np.zeros(N_TIME_BINS - duration_bins)])


HALFNORMAL = sm.build_cell_generation_prob(DURATION)
MATCHED_PROB = HALFNORMAL.sum() / DURATION  # keeps the expected total number of generation events

# The order defines the random seeds; keep it unchanged for reproducibility.
GENERATION_CONDITIONS = {
    "halfnormal_34h": HALFNORMAL,
    "flat_matched_34h": build_flat_prob(MATCHED_PROB, DURATION),
    "flat_p1_34h": build_flat_prob(1.0, DURATION),
}
GRID_ORDER = ["cyclerandomness", "cycle_diff", "trapeffectiveheight",
              "tangentialmigrationcellcycle", "tangentialmigrationprob", "IPratio"]


def build_lattice():
    positions, _ = sm.generate_lattice_positions()
    tree = cKDTree(positions)
    dist, _ = tree.query(positions, k=2)
    min_distance = dist[:, 1].min()
    neighbor_lists = [[j for j in tree.query_ball_point(p, r=1.1 * min_distance) if j != i]
                      for i, p in enumerate(positions)]
    n_neighbors = np.array([len(n) for n in neighbor_lists])
    neighbors = np.full((len(positions), n_neighbors.max()), -1, dtype=int)
    for i, n in enumerate(neighbor_lists):
        neighbors[i, :len(n)] = n
    return positions, neighbors, n_neighbors


POSITIONS, NEIGHBORS, N_NEIGHBORS = build_lattice()
N_FIBER = len(POSITIONS)


def initial_cell_cycle(rng, cycle_randomness, cycle_diff):
    cellcycle = np.empty(N_FIBER)
    cellcycle[:N_TYPE0] = 1 - cycle_randomness * 2 - cycle_randomness * rng.random(N_TYPE0)
    cellcycle[N_TYPE0:] = 1 - cycle_randomness * 2 - cycle_diff - cycle_randomness * rng.random(N_FIBER - N_TYPE0)
    return cellcycle - np.floor(cellcycle)


def simulation(rng, cellcycle, trap_effective_height, tangential_migration_cell_cycle,
               tangential_migration_prob, IP_ratio, cell_cycle_hours, cell_generation_prob):
    neuron_fiber = np.empty(0, dtype=int)
    neuron_position = np.empty(0, dtype=int)
    trap = np.zeros(N_FIBER, dtype=bool)
    cellcycle = cellcycle.copy()
    step = 1 / cell_cycle_hours / sm.BINS_PER_HOUR

    for timebin in range(N_TIME_BINS):
        trap[neuron_fiber[neuron_position >= trap_effective_height + sm.IMZ_END]] = True

        move = ((tangential_migration_prob > rng.random(len(neuron_fiber))) &
                (cellcycle[neuron_fiber] <= tangential_migration_cell_cycle) &
                (neuron_position <= sm.IMZ_END) &
                (neuron_position >= sm.IMZ_START) &
                ~trap[neuron_fiber])
        moving_fiber = neuron_fiber[move]
        choice = (rng.random(len(moving_fiber)) * N_NEIGHBORS[moving_fiber]).astype(int)
        neuron_fiber[move] = NEIGHBORS[moving_fiber, choice]

        neuron_position += 1

        cellcycle += step
        cellcycle -= np.floor(cellcycle)
        dividing = np.where(cellcycle < step)[0]
        generating = dividing[cell_generation_prob[timebin] > rng.random(len(dividing))]
        via_ip = IP_ratio > rng.random(len(generating))
        new_fiber = np.concatenate([generating, generating[via_ip]])
        neuron_fiber = np.concatenate([neuron_fiber, new_fiber])
        neuron_position = np.concatenate([neuron_position, np.zeros(len(new_fiber), dtype=int)])

    return np.bincount(neuron_fiber, minlength=N_FIBER)


def seed_entropy(base_seed, cond_index, seed_index, param_index):
    # base seed 0 keeps the seeds of the published run
    if base_seed == 0:
        return [cond_index, seed_index, param_index]
    return [base_seed, cond_index, seed_index, param_index]


def run_one(task):
    generation_name, seed_index, base_seed, seed, params = task
    rng = np.random.default_rng(seed)
    cellcycle = initial_cell_cycle(rng, params["cyclerandomness"], params["cycle_diff"])
    cell_count = simulation(
        rng, cellcycle,
        params["trapeffectiveheight"],
        params["tangentialmigrationcellcycle"],
        params["tangentialmigrationprob"],
        params["IPratio"],
        sm.CELL_CYCLE_HOURS,
        GENERATION_CONDITIONS[generation_name],
    )
    mean_type0 = cell_count[:N_TYPE0].mean()
    mean_type1 = cell_count[N_TYPE0:].mean()
    return {
        "generation": generation_name,
        "seed_index": seed_index,
        "base_seed": base_seed,
        **params,
        "n_neurons": int(cell_count.sum()),
        "celltype_ratio": mean_type1 / mean_type0 if mean_type0 > 0 else np.inf,
    }


def bias(ratio):
    ratio = np.asarray(ratio, dtype=float)
    return np.maximum(ratio, 1 / ratio)


HIST_BINS = np.arange(0.5, 13.5, 1.0)


def plot_summary(df, path):
    names = [n for n in GENERATION_CONDITIONS if n in set(df["generation"])]
    seeds = sorted(df["seed_index"].unique())
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6), dpi=200,
                             gridspec_kw={"height_ratios": [1, 2.5]}, constrained_layout=True)
    time_axis = np.arange(N_TIME_BINS) / sm.BINS_PER_HOUR
    for col, name in enumerate(names):
        ax = axes[0, col]
        ax.plot(time_axis, HALFNORMAL, ":", color="lightgray", lw=2)
        ax.plot(time_axis, GENERATION_CONDITIONS[name], "-", color="tab:orange", lw=2)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xticks([0, 24, 48, 72, 96, 120])
        ax.set_xlabel("time (hours)", fontsize=8)
        ax.set_title(name, fontsize=9)
        if col == 0:
            ax.set_ylabel("Proportion of\ncell generation", fontsize=8)

        ax = axes[1, col]
        sub = df[df["generation"] == name]
        # bar: mean over seeds, orange tick: each seed
        counts = np.array([np.histogram(bias(sub[sub["seed_index"] == s]["celltype_ratio"]), bins=HIST_BINS)[0]
                           for s in seeds])
        centers = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2
        ax.bar(centers, counts.mean(axis=0), width=1.0, color="#3f0f6e", edgecolor="gray")
        for c in counts:
            ax.plot(centers, c, "_", color="tab:orange", ms=10, mew=1.5)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 2304)
        ax.set_xlabel("Ratio of # neurons in the two kinds of fibers\n(larger / smaller)", fontsize=7)
        ax.text(0.95, 0.95, f"mean #neurons: {sub['n_neurons'].mean():.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
        if col == 0:
            ax.set_ylabel("Frequency", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="number of independent repetitions")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0, help="base random seed (default 0 = published run)")
    parser.add_argument("--random-seed", action="store_true", help="use a random base seed")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"matched flat probability: {MATCHED_PROB:.4f}")
    base_seed = sm.resolve_seed(None if args.random_seed else args.seed)

    param_list = [dict(zip(GRID_ORDER, values))
                  for values in itertools.product(*[sm.PARAMETER_GRID[k] for k in GRID_ORDER])]
    tasks = []
    for cond_index, name in enumerate(GENERATION_CONDITIONS):
        for seed_index in range(args.seeds):
            for param_index, params in enumerate(param_list):
                tasks.append((name, seed_index, base_seed,
                              seed_entropy(base_seed, cond_index, seed_index, param_index), params))

    with Pool(args.workers) as pool:
        records = list(pool.imap_unordered(run_one, tasks, chunksize=16))

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.out_dir, "ed_fig6a-d_flat_generation.csv")
    df.to_csv(csv_path, index=False)
    plot_summary(df, os.path.join(args.out_dir, "ed_fig6a-d_flat_generation.png"))
    print("saved", csv_path)


if __name__ == "__main__":
    main()
