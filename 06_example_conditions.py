import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulation_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {
    "condition1": {"cycle_diff": 0.5, "cyclerandomness": 0.1, "IPratio": 0.1,
                   "tangentialmigrationprob": 0.5, "tangentialmigrationcellcycle": 0.5,
                   "trapeffectiveheight": 0},
    "condition2": {"cycle_diff": 0.01, "cyclerandomness": 0.1, "IPratio": 0.1,
                   "tangentialmigrationprob": 1 / 36, "tangentialmigrationcellcycle": 0.2,
                   "trapeffectiveheight": 0},
}
BROWN = "#8c510a"
CYAN = "#35b6c8"


def run_condition(name, params, seed, out_dir):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    radial_fiber_list = sm.define_radial_fibers(params["cyclerandomness"], params["cycle_diff"], rng=rng)
    cell_count, neuron_list, cc_history = sm.simulation(
        radial_fiber_list,
        params["trapeffectiveheight"],
        params["tangentialmigrationcellcycle"],
        params["tangentialmigrationprob"],
        params["IPratio"],
        record_cell_cycle=True,
    )
    ratio = sm.celltype_ratio(cell_count)
    fiber_xy, neuron_xy, neuron_fiber_ids = sm.scaled_coordinates(radial_fiber_list, neuron_list, rng)
    _, normalized_density = sm.relative_density(fiber_xy, neuron_xy)
    types = np.array([rf.type for rf in radial_fiber_list])
    positions = np.array([rf.position for rf in radial_fiber_list])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=200, constrained_layout=True)

    # cell cycle of the two populations (first 48 h, 5 fibers each)
    ax = axes[0]
    t = np.arange(1, sm.N_TIME_BINS + 1) / sm.BINS_PER_HOUR
    shown = t <= 48
    for k in np.where(types == 0)[0][:5]:
        ax.plot(t[shown], cc_history[shown, k], color=BROWN, lw=1)
    for k in np.where(types == 1)[0][:5]:
        ax.plot(t[shown], cc_history[shown, k], color=CYAN, lw=1)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cell cycle state")
    ax.set_title("Cell cycle (brown: type 0, cyan: type 1)", fontsize=9)

    # number of SCPNs on each fiber
    ax = axes[1]
    sc = ax.scatter(positions[:, 0], positions[:, 1], c=cell_count, cmap="viridis", s=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"# SCPNs per fiber (ratio type1/type0 = {ratio:.2f})", fontsize=9)
    fig.colorbar(sc, ax=ax, shrink=0.8)

    # rescaled coordinates (central 120 um x 120 um)
    ax = axes[2]
    center = fiber_xy.mean(axis=0)
    half = 60
    for xy, kw in [(fiber_xy[types == 0], dict(color=BROWN, s=10)),
                   (fiber_xy[types == 1], dict(color=CYAN, s=10)),
                   (neuron_xy, dict(color="limegreen", s=25, alpha=0.7))]:
        inside = np.all(np.abs(xy - center) <= half, axis=1)
        ax.scatter(xy[inside, 0], xy[inside, 1], **kw)
    ax.set_aspect("equal")
    ax.set_xlabel("um")
    ax.set_title("Rescaled coordinates (green: SCPNs)", fontsize=9)

    # relative density
    ax = axes[3]
    centers = (sm.DISTANCE_BIN_EDGES[:-1] + sm.DISTANCE_BIN_EDGES[1:]) / 2
    ax.plot(centers, normalized_density, "o-", color="limegreen")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Tangential distance (um)")
    ax.set_ylabel("Normalized relative density")
    ax.set_title("Relative density of SCPNs", fontsize=9)

    fig.suptitle(name)
    fig.savefig(os.path.join(out_dir, f"example_{name}.png"))
    fig.savefig(os.path.join(out_dir, f"example_{name}.eps"))
    plt.close(fig)
    print(f"{name}: ratio = {ratio:.3f}, isolated fiber ratio = {sm.isolated_fiber_ratio(cell_count):.3f}, "
          f"0-3 um density = {normalized_density[0]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed = sm.resolve_seed(args.seed)
    for name, params in CONDITIONS.items():
        run_condition(name, params, seed, args.out_dir)


if __name__ == "__main__":
    main()
