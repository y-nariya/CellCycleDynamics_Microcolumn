import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTAL_ISOLATED_FIBER_RATIO = 0.2832              # mean of three embryos (Fig. 6 data)
EXPERIMENTAL_DENSITY_0_3UM = [1.4807, 1.4708, 1.6428]  # individual embryos (Fig. 4i)

DISTANCE_BINS = ["0.0_3.0um", "3.0_6.0um", "6.0_9.0um", "9.0_12.0um", "12.0_15.0um",
                 "15.0_18.0um", "18.0_21.0um", "21.0_24.0um", "24.0_27.0um", "27.0_30.0um"]
DENSITY_COLS = [f"normalized_cell_density_{b}" for b in DISTANCE_BINS]
BAND_LOWER = np.arange(1.0, 4.0, 0.25)
BAND_UPPER = BAND_LOWER + 0.25
BAND_MID = BAND_LOWER + 0.125


def band_summary(df):
    ratio = df["celltype_ratio"]
    rows = []
    for x, upper in zip(BAND_LOWER, BAND_UPPER):
        mask = ((ratio >= x) & (ratio <= upper)) | ((ratio >= 1.0 / upper) & (ratio <= 1.0 / x))
        sub = df[mask]
        row = {"band_lower": x, "band_upper": upper, "n": len(sub),
               "isolated_mean": sub["isolated_fiber_ratio"].mean(),
               "isolated_std": sub["isolated_fiber_ratio"].std()}
        for col, b in zip(DENSITY_COLS, DISTANCE_BINS):
            row[f"density_mean_{b}"] = sub[col].mean()
            row[f"density_std_{b}"] = sub[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(HERE, "data", "grid_search_results.csv"))
    parser.add_argument("--subset", choices=["all", "manuscript"], default="all")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.subset == "manuscript":
        df = df[(df["cell_generation_mode"] == "short") & (df["CellCycleHour"] == 12)]
    print(f"{len(df)} simulations ({args.subset})")

    summary = band_summary(df)
    summary.to_csv(os.path.join(args.out_dir, f"ed_fig7e-f_band_summary_{args.subset}.csv"), index=False)

    valid = summary["isolated_mean"].notna().to_numpy()
    iso_mean = summary["isolated_mean"].to_numpy()
    d0_mean = summary["density_mean_0.0_3.0um"].to_numpy()

    predicted_ratio = float(interp1d(iso_mean[valid], BAND_MID[valid],
                                     kind="linear", fill_value="extrapolate")(EXPERIMENTAL_ISOLATED_FIBER_RATIO))
    predicted_density = float(interp1d(BAND_LOWER[valid], d0_mean[valid],
                                       kind="linear", fill_value="extrapolate")(predicted_ratio))
    print(f"isolated fiber ratio {EXPERIMENTAL_ISOLATED_FIBER_RATIO} -> allocation ratio {predicted_ratio:.4f}")
    print(f"predicted 0-3 um normalized relative density: {predicted_density:.4f}")
    print(f"experimental values: {EXPERIMENTAL_DENSITY_0_3UM}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200, constrained_layout=True)
    tick_labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(BAND_LOWER, BAND_UPPER)]

    # (e) allocation ratio band vs isolated fiber ratio
    ax = axes[0]
    ax.errorbar(BAND_MID, iso_mean, yerr=summary["isolated_std"], fmt="o-", color="magenta", capsize=4, lw=2)
    ax.annotate("", xy=(BAND_MID[0] - 0.2, EXPERIMENTAL_ISOLATED_FIBER_RATIO),
                xytext=(BAND_MID[0] - 0.45, EXPERIMENTAL_ISOLATED_FIBER_RATIO),
                arrowprops=dict(arrowstyle="-|>", color="yellowgreen", lw=2))
    ax.axhline(EXPERIMENTAL_ISOLATED_FIBER_RATIO, color="yellowgreen", ls=":", lw=1)
    ax.axvline(predicted_ratio, color="yellowgreen", ls=":", lw=1.5)
    ax.set_xticks(BAND_MID)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Ratio of # SCPNs in the two kinds of fibers")
    ax.set_ylabel("Isolated fiber ratio")
    ax.set_ylim(0, 1)
    ax.text(-0.12, 1.06, "e", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # (f) allocation ratio band vs normalized relative density in each distance bin
    ax = axes[1]
    cmap = plt.cm.viridis
    for j, b in enumerate(DISTANCE_BINS):
        color = cmap(j / (len(DISTANCE_BINS) - 1))
        m = summary[f"density_mean_{b}"].to_numpy()
        s = summary[f"density_std_{b}"].to_numpy()
        ax.plot(BAND_LOWER, m, "-", color=color, lw=1.5, label=b.replace("_", "-"))
        ax.fill_between(BAND_LOWER, m - s, m + s, color=color, alpha=0.12)
    for v in EXPERIMENTAL_DENSITY_0_3UM:
        ax.annotate("", xy=(BAND_LOWER[0] - 0.05, v), xytext=(BAND_LOWER[0] - 0.3, v),
                    arrowprops=dict(arrowstyle="-|>", color="cyan", lw=1.5))
    ax.annotate("", xy=(BAND_LOWER[0] - 0.05, predicted_density), xytext=(BAND_LOWER[0] - 0.3, predicted_density),
                arrowprops=dict(arrowstyle="-|>", color="yellowgreen", lw=2))
    ax.axvline(predicted_ratio, color="yellowgreen", ls=":", lw=1.5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(BAND_LOWER)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Ratio of # SCPNs in the two kinds of fibers")
    ax.set_ylabel("Normalized relative density")
    ax.legend(fontsize=6, ncol=2, title="Tangential distance", title_fontsize=6)
    ax.text(-0.12, 1.06, "f", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    stem = os.path.join(args.out_dir, f"ed_fig7e-f_isolated_fiber_prediction_{args.subset}")
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".eps")
    plt.close(fig)
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
