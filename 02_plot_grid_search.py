import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulation_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))

# (column, title, x-axis note)
PANELS = [
    ("cycle_diff", "(1) Phase difference\nof cell cycle", "Small ← → Large"),
    ("cyclerandomness", "(2) Phase variation\nof cell cycle", "Small ← → Large"),
    ("IPratio", "(3) Generation ratio\nof IP", "More IP →"),
    ("tangentialmigrationprob", "(4) Frequency of\ntangential migration", "Low ← → High"),
    ("tangentialmigrationcellcycle", "(5) Phase specificity of\ntangential migration", "Specific ←"),
    ("trapeffectiveheight", "(6) Time of\naxon trapping", "Time after generation"),
]
PANEL_LETTERS = "jklmno"
# trapeffectiveheight (bins) -> trapping starts (108 + value) / 3 h after generation; 360 = no trapping
TRAP_LABELS = {0: "36 h", 36: "48 h", 360: "None"}
Y_LABEL = "Ratio of # neurons in the two kinds of fibers\n(larger / smaller)"
YLIM = (1, 12)


def upper_tail(values, stat):
    if stat == "q95":
        return values.quantile(0.95)
    # mean of the top 5% of conditions
    return values[values >= values.quantile(0.95)].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(HERE, "data", "grid_search_results.csv"))
    parser.add_argument("--stat", choices=["q95", "top5_mean"], default="top5_mean",
                        help="blue line: 95th percentile or mean of the top 5%% per parameter value")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Fig. 3d ----
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200, constrained_layout=True)
    t = np.arange(sm.N_TIME_BINS) / sm.BINS_PER_HOUR
    ax.plot(t, sm.CELL_GENERATION_PROB, "-", color="black", lw=2)
    ax.set_xlim(0, sm.TOTAL_HOURS)
    ax.set_xticks([0, 24, 48, 72, 96, 120])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Probability of SCPN generation")
    fig.savefig(os.path.join(args.out_dir, "fig3d_generation_probability.png"))
    fig.savefig(os.path.join(args.out_dir, "fig3d_generation_probability.eps"))
    plt.close(fig)

    rng = np.random.default_rng(0)  # horizontal jitter of the dots
    df = pd.read_csv(args.csv)
    df = df[(df["cell_generation_mode"] == "short") & (df["CellCycleHour"] == 12)].copy()
    df["bias"] = np.maximum(df["celltype_ratio"], 1.0 / df["celltype_ratio"])
    print(f"{len(df)} simulations (short generation, 12 h cell cycle)")

    # ---- Fig. 3e ----
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=200, constrained_layout=True)
    ax.hist(df["bias"], bins=np.arange(1, 12.5, 0.5), color="#3f0f6e", edgecolor="gray")
    ax.set_xlabel(Y_LABEL, fontsize=8)
    ax.set_ylabel("Frequency")
    fig.savefig(os.path.join(args.out_dir, "fig3e_ratio_histogram.png"))
    fig.savefig(os.path.join(args.out_dir, "fig3e_ratio_histogram.eps"))
    plt.close(fig)

    # ---- Fig. 3j-o ----
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=200, sharey=True, constrained_layout=True)
    for ax, letter, (key, title, caption) in zip(axes.flat, PANEL_LETTERS, PANELS):
        sub = df[[key, "bias"]].dropna()
        levels = sorted(sub[key].unique())
        stat = sub.groupby(key)["bias"].apply(lambda s: upper_tail(s, args.stat)).reindex(levels)

        if key == "trapeffectiveheight":
            # 0, 36, 360 are far apart -> equally spaced categories with an axis break before the last one
            x = sub[key].map({v: i for i, v in enumerate(levels)}).to_numpy() + rng.uniform(-0.08, 0.08, len(sub))
            x_line = range(len(levels))
            ticks = list(range(len(levels)))
            tick_labels = [TRAP_LABELS.get(int(v), str(v)) for v in levels]
            ax.text(len(levels) - 1.5, -0.06, "//", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=11)
        else:
            jitter = (levels[-1] - levels[0]) * 0.01
            x = sub[key].to_numpy() + rng.uniform(-jitter, jitter, len(sub))
            x_line = levels
            ticks = levels
            tick_labels = [f"{v:.2f}" if abs(v - round(v)) > 1e-6 else f"{int(v)}" for v in levels]

        ax.plot(x, sub["bias"], ".", color="gray", ms=2, alpha=0.15, zorder=1)
        ax.plot(x_line, stat.values, "o-", color="tab:blue", lw=2, ms=6, zorder=3)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_title(title, fontsize=10)
        ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")
        ax.set_xlabel(caption, fontsize=8)
        ax.set_ylim(*YLIM)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL, fontsize=9)

    fig.savefig(os.path.join(args.out_dir, "fig3j-o_parameter_dependence.png"))
    fig.savefig(os.path.join(args.out_dir, "fig3j-o_parameter_dependence.eps"))
    plt.close(fig)
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
