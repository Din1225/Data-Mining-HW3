"""Draw paired effects and rank movements from verified full-catalogue ledgers."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run import ROOT

DEST = ROOT / "results/comparison"
SPLITS = (("test", "2016"), ("prospective", "2018"))
PRIMARY = "adaptive_graph_creator"
COMPARATORS = (
    ("adaptive_graph", "Parent graph"),
    ("fixed", "Fixed fusion"),
    ("two_year", "Two-year popularity"),
    ("als_blend", "ALS + popularity"),
    ("ingredients_idf", "Weighted ingredients"),
)
COLORS = ("#165481", "#AA4D18")


def paired_effects(primary, comparators, dest):
    fig, ax = plt.subplots(figsize=(3.35, 1.98))
    y = np.arange(len(comparators))[::-1]
    for index, (split, year) in enumerate(SPLITS):
        records = pd.read_csv(DEST / f"{split}_{primary}_contrasts.csv")
        records = records[records.metric.eq("NDCG@10")].set_index("method")
        rows = records.loc[[method for method, _ in comparators]]
        estimates = rows.delta.to_numpy() * 100
        low, hi = rows.user_lo.to_numpy() * 100, rows.user_hi.to_numpy() * 100
        xpos = y + (0.13 if index == 0 else -0.13)
        ax.errorbar(
            estimates,
            xpos,
            xerr=np.vstack([estimates - low, hi - estimates]),
            color=COLORS[index],
            markersize=3.2,
            fmt=("o", "s")[index],
            lw=0.85,
            capsize=1.8,
            elinewidth=1.0,
            label=year,
            zorder=3,
        )
    ax.axvline(0, color="#565656", linestyle=":", linewidth=0.75)
    ax.set_yticks(y, [label for _, label in comparators])
    ax.set_ylim(-0.45, len(y) - 0.55)
    ax.set_xlabel("Paired difference in NDCG@10 (×100)")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.4,
        columnspacing=1.2,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.35, color="#D6D8DA")
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.34, right=0.985, bottom=0.215, top=0.84)
    fig.savefig(dest / "paired_effects.pdf", metadata={"Title": "Paired NDCG@10 differences"})
    fig.savefig(dest / "paired_effects.png", dpi=240)
    plt.close(fig)


def creator_movements(primary, dest):
    fig, ax = plt.subplots(figsize=(3.35, 2.00))
    colors = ("#146687", "#D3D7D8", "#BC633F")
    ypos = np.array([5.4, 4.4, 3.4, 1.95, 0.95, -0.05])
    labels, strata = [], []
    for split, year in SPLITS:
        data = pd.read_csv(DEST / f"{split}_{primary}_creator_changes.csv")
        rows = data[data.k.eq(10)].set_index("creator_prior_likes")
        for exposure in ("0", "1", "2+"):
            row = rows.loc[exposure]
            strata.append([row.improved / row.users, row.tied / row.users, row.hurt / row.users])
            labels.append(f"{exposure} ({int(row.users):,})")
    scores = 100 * np.array(strata)
    for idx, (word, color) in enumerate(zip(("Higher", "Tied", "Lower"), colors)):
        left = scores[:, :idx].sum(axis=1)
        ax.barh(ypos, scores[:, idx], left=left, height=0.64, color=color, label=word)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, 6.05)
    ax.set_yticks(ypos, labels)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Users by change in target rank (%)")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        columnspacing=1.05,
        handlelength=1.1,
        handletextpad=0.35,
    )
    for year, y in (("2016", 4.4), ("2018", 0.95)):
        ax.text(
            -0.32,
            y,
            year,
            ha="right",
            va="center",
            fontsize=7,
            fontweight="bold",
            transform=ax.get_yaxis_transform(),
        )
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.21, top=0.84)
    fig.savefig(
        dest / "creator_rank_movements.pdf",
        metadata={"Title": "Target rank by creator familiarity"},
    )
    fig.savefig(dest / "creator_rank_movements.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default=PRIMARY)
    parser.add_argument("--output", type=Path, default=DEST / "figures")
    parser.add_argument("--compare", nargs=2, metavar=("METHOD", "LABEL"), action="append")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "legend.fontsize": 7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "pdf.fonttype": 42,
        }
    )
    paired_effects(args.primary, args.compare or COMPARATORS, args.output)
    creator_movements(args.primary, args.output)


if __name__ == "__main__":
    main()
