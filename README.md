# Simulation code for "Cell cycle dynamics of radial glia direct neocortical module formation"

Python code for the simulation of SCPN allocation to radial fibers
(Fig. 3, Extended Data Fig. 6, Extended Data Fig. 7).

Tested with Python 3.13, numpy 2.1, scipy 1.15, pandas 2.2, matplotlib 3.10.

```
pip install -r requirements.txt
```

## Files

| File | Content | Figure |
|---|---|---|
| `simulation_model.py` | Model (hexagonal lattice of 902 RGCs, cell cycle, neuron generation, tangential migration, axon trapping), rescaling to experimental coordinates, relative density | – |
| `01_run_grid_search.py` | Runs the 2,304 parameter combinations (`--full`: 36,864, short/long generation × 8 cell cycle durations) | Fig. 3e, j–o; ED Fig. 7 |
| `02_plot_grid_search.py` | SCPN generation probability; histogram of the allocation ratio and its dependence on each parameter | Fig. 3d, e, j–o |
| `03_flat_generation_probability.py` | Flat SCPN generation probability vs the original half-normal one | ED Fig. 6a–d |
| `04_hexagonal_order_vs_bias.py` | Progressive randomization of the initial phase assignment; hexatic order (mean \|ψ6\|) vs CV / accumulation ratio | ED Fig. 6e, f |
| `05_isolated_fiber_prediction.py` | Allocation ratio vs isolated fiber ratio and normalized relative density; prediction from the experimental isolated fiber ratio | ED Fig. 7e, f |
| `06_example_conditions.py` | The two example conditions (cell cycle traces, SCPNs per fiber, rescaled coordinates, relative density) | Fig. 3f–i; ED Fig. 7a–d |

## Reproduction

The grid-search CSV is not included in this repository. Generate it locally before
running the grid-search plotting and prediction scripts:

```
python 01_run_grid_search.py --full --out results/grid_search.csv
python 02_plot_grid_search.py --csv results/grid_search.csv  # Fig. 3d, e, j-o
python 05_isolated_fiber_prediction.py --csv results/grid_search.csv  # ED Fig. 7e, f
python 04_hexagonal_order_vs_bias.py        # ED Fig. 6e, f (r = 0.91 and r = 0.43), ~4 min
python 03_flat_generation_probability.py    # ED Fig. 6a-d (5 repetitions), ~1 min with 10 cores
python 06_example_conditions.py             # Fig. 3f-i, ED Fig. 7a-d
```

### Random seeds

`03` and `04` use base seed 0 by default and reproduce the published results exactly;
`--random-seed` runs them with a random base seed instead.
`01` and `06` use a random base seed unless `--seed` is given. The original grid search was run
without fixed seeds. A regenerated grid therefore agrees
statistically but not run by run. Pass `--seed N` to make a regenerated grid repeatable.

## Parameters

| Parameter (column name) | Values |
|---|---|
| Phase difference between the two RGC populations (`cycle_diff`) | 0.01, 0.1, 0.3, 0.5 |
| Phase variation within each population (`cyclerandomness`) | 0.01, 0.1, 0.3, 0.5 |
| Generation ratio of intermediate progenitors (`IPratio`) | 0.1, 0.3, 0.5 |
| Probability of tangential migration per 20-min bin (`tangentialmigrationprob`) | 1/36, 1/12, 1/6, 0.5 |
| Cell cycle threshold for tangential migration (`tangentialmigrationcellcycle`) | 0.2, 0.5, 0.8, 1.0 |
| Axon trapping (`trapeffectiveheight`, bins) | 0, 36, 360 (= trapping from 36 h, 48 h after generation, none) |

Fixed: 120 h, 360 bins, cell cycle 12 h, 18 × 18 lattice (324 + 578 RGCs), tangential migration
24–36 h after generation, half-normal generation probability over 101 bins (~34 h).

## License

MIT License (see `LICENSE`).
