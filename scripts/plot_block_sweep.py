#!/usr/bin/env python3
"""
plot_block_sweep.py — Plot GFLOP/s vs BLOCK_NB from the panel-width sweep.

Usage:
    python3 scripts/plot_block_sweep.py

Input:   results/block_sweep.csv
Output:  report/figures/fig8_block_sweep.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(ROOT, "results", "block_sweep.csv")
OUT_PDF = os.path.join(ROOT, "report", "figures", "fig8_block_sweep.pdf")
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

df = pd.read_csv(IN_CSV)

# Aggregate reps → mean ± std per BLOCK_NB
agg = df.groupby("BLOCK_NB").agg(
    gflops_mean=("gflops", "mean"),
    gflops_std =("gflops", "std"),
    time_mean  =("time_s", "mean"),
).reset_index().sort_values("BLOCK_NB")

# Pull metadata for the title
n       = int(df["n"].iloc[0])
threads = int(df["threads"].iloc[0])

fig, ax = plt.subplots(figsize=(5.5, 4))

ax.errorbar(
    agg["BLOCK_NB"], agg["gflops_mean"],
    yerr=agg["gflops_std"],
    marker="o", linewidth=1.8, capsize=4,
    color="#9467bd", label="Mean ± 1 SD",
)

# Mark the best NB
best_idx = agg["gflops_mean"].idxmax()
best_nb  = int(agg.loc[best_idx, "BLOCK_NB"])
best_gf  = float(agg.loc[best_idx, "gflops_mean"])
ax.axvline(best_nb, color="red", linestyle="--", linewidth=1.0,
           label=f"Best NB={best_nb} ({best_gf:.1f} GFLOP/s)")

ax.set_xlabel("Panel width BLOCK_NB (doubles)")
ax.set_ylabel("Performance (GFLOP/s)")
ax.set_title(
    f"Fig 8 — GFLOP/s vs panel width (v5_openmp_blocked)\n"
    f"n={n}, {threads} threads, CSD3 icelake, {len(df['rep'].unique())} reps"
)
ax.set_xticks(agg["BLOCK_NB"].tolist())

# Annotate L1 cache capacity: 128 doubles = 1 KB, L1 = 32 KB → 4 rows of 128
ax.text(128, ax.get_ylim()[0], "  L1-fit\n  (128×8=1KB)", fontsize=7,
        color="gray", va="bottom")

ax.legend(fontsize=9)
ax.grid(True, alpha=0.35, linestyle="--")

fig.tight_layout()
fig.savefig(OUT_PDF)
print(f"Saved {OUT_PDF}")
