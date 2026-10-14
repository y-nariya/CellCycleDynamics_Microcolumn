import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import simulation_model as sm

# Generation probability time courses ("short" is the one used in the manuscript, Fig. 3d)
GENERATION_MODES = {
    "short": sm.build_cell_generation_prob(sm.GENERATION_DURATION_BINS),   # ~34 h
    "long": sm.build_cell_generation_prob(72 * sm.BINS_PER_HOUR),          # 72 h
}
FULL_CELL_CYCLE_HOURS = [9, 12, 13, 15, 17, 18, 20, 22]

BIN_LABELS = [f"{a:.1f}_{b:.1f}um" for a, b in zip(sm.DISTANCE_BIN_EDGES[:-1], sm.DISTANCE_BIN_EDGES[1:])]


def run_one(task):
    index, seed, mode, cell_cycle_hours, params = task
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    radial_fiber_list = sm.define_radial_fibers(params["cyclerandomness"], params["cycle_diff"], rng=rng)
    cell_count, neuron_list = sm.simulation(
        radial_fiber_list,
        params["trapeffectiveheight"],
        params["tangentialmigrationcellcycle"],
        params["tangentialmigrationprob"],
        params["IPratio"],
        cell_cycle_hours,
        GENERATION_MODES[mode],
    )
    fiber_xy, neuron_xy, _ = sm.scaled_coordinates(radial_fiber_list, neuron_list, rng)
    mean_count, normalized_density = sm.relative_density(fiber_xy, neuron_xy)

    row = {
        "index": index,
        "seed": seed,
        "cell_generation_mode": mode,
        **params,
        "CellCycleHour": cell_cycle_hours,
        "n_neurons": len(neuron_list),
        "celltype_ratio": sm.celltype_ratio(cell_count),
        "isolated_fiber_ratio": sm.isolated_fiber_ratio(cell_count),
    }
    row.update({f"mean_neuron_count_{b}": v for b, v in zip(BIN_LABELS, mean_count)})
    row.update({f"normalized_cell_density_{b}": v for b, v in zip(BIN_LABELS, normalized_density)})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="short/long generation x 8 cell cycle durations (36,864 combinations)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=None, help="base random seed (default: random)")
    parser.add_argument("--out", default=os.path.join("results", "grid_search.csv"))
    args = parser.parse_args()

    modes = list(GENERATION_MODES) if args.full else ["short"]
    cell_cycle_hours_list = FULL_CELL_CYCLE_HOURS if args.full else [sm.CELL_CYCLE_HOURS]
    keys = list(sm.PARAMETER_GRID)
    param_list = [dict(zip(keys, values)) for values in itertools.product(*sm.PARAMETER_GRID.values())]

    base_seed = sm.resolve_seed(args.seed)
    tasks = []
    for mode, cch in itertools.product(modes, cell_cycle_hours_list):
        for params in param_list:
            index = len(tasks)
            tasks.append((index, base_seed + index, mode, cch, params))
    print(f"{len(tasks)} simulations")

    records = []
    with Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4)):
            records.append(rec)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(tasks)} done", flush=True)

    df = pd.DataFrame(records).sort_values("index")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
