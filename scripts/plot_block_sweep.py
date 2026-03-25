#!/usr/bin/env python3
"""
plot_block_sweep.py — Plot GFLOP/s vs BLOCK_NB from the panel-width sweep.

Usage:
    python3 scripts/plot_block_sweep.py

Input:   results/block_sweep.csv
Output:  report/figures/fig8_block_sweep.pdf
         results/block_sweep_summary.csv
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.environ.get(
    "BLOCK_SWEEP_CSV",
    os.path.join(ROOT, "results", "block_sweep.csv"),
)
OUT_PDF = os.environ.get(
    "BLOCK_SWEEP_FIG",
    os.path.join(ROOT, "report", "figures", "fig8_block_sweep.pdf"),
)
SUMMARY_CSV = os.environ.get(
    "BLOCK_SWEEP_SUMMARY_CSV",
    os.path.join(ROOT, "results", "block_sweep_summary.csv"),
)
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)

df = pd.read_csv(IN_CSV)
has_version_col = "version" in df.columns

# Backward compatibility: old files omitted "version" column.
if not has_version_col:
    df["version"] = "blocked_version_unspecified"

versions = sorted(df["version"].unique())
ns = sorted(df["n"].unique())

if len(versions) != 1:
    raise SystemExit(
        f"{IN_CSV} contains multiple versions {versions}. "
        "Run one block sweep per version so the figure stays interpretable."
    )
if len(ns) != 1:
    raise SystemExit(
        f"{IN_CSV} contains multiple matrix sizes {ns}. "
        "Run one block sweep per matrix size so the figure stays interpretable."
    )

# Aggregate reps → mean ± std per (threads, BLOCK_NB)
agg = df.groupby(["threads", "BLOCK_NB"]).agg(
    gflops_mean=("gflops", "mean"),
    gflops_std =("gflops", "std"),
    time_mean  =("time_s", "mean"),
).reset_index().sort_values(["threads", "BLOCK_NB"])

# Pull metadata for the title
version = str(versions[0])
n = int(ns[0])
thread_values = sorted(int(t) for t in agg["threads"].unique())
block_sizes = sorted(int(nb) for nb in agg["BLOCK_NB"].unique())
reps = len(df["rep"].unique())

# For each thread count, keep the fastest BLOCK_NB. Ties prefer the smaller NB.
summary = (
    agg.sort_values(["threads", "gflops_mean", "BLOCK_NB"], ascending=[True, False, True])
       .groupby("threads", as_index=False)
       .first()
       .rename(columns={
           "BLOCK_NB": "best_block_nb",
           "gflops_mean": "best_gflops_mean",
           "gflops_std": "best_gflops_std",
           "time_mean": "best_time_mean",
       })
)
summary.insert(0, "version", version)
summary.insert(1, "n", n)
summary["reps"] = reps
summary["num_block_sizes"] = summary["threads"].map(
    df.groupby("threads")["BLOCK_NB"].nunique()
)
summary.to_csv(SUMMARY_CSV, index=False)

if len(thread_values) == 1:
    threads = thread_values[0]
    curve = agg[agg["threads"] == threads].sort_values("BLOCK_NB")
    best = summary.iloc[0]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(
        curve["BLOCK_NB"], curve["gflops_mean"],
        yerr=curve["gflops_std"],
        marker="o", linewidth=1.8, capsize=4,
        color="#9467bd", label="Mean ± 1 SD",
    )
    ax.axvline(
        int(best["best_block_nb"]),
        color="red", linestyle="--", linewidth=1.0,
        label=f"Best NB={int(best['best_block_nb'])} ({float(best['best_gflops_mean']):.1f} GFLOP/s)",
    )
    ax.set_xlabel("Panel width BLOCK_NB (doubles)")
    ax.set_ylabel("Performance (GFLOP/s)")
    if has_version_col:
        ax.set_title(
            f"GFLOP/s vs panel width\n"
            f"{version}, n={n}, {threads} threads, CSD3 icelake, {reps} reps"
        )
    else:
        ax.set_title(
            f"GFLOP/s vs panel width\n"
            f"n={n}, {threads} threads, CSD3 icelake, {reps} reps"
        )
    ax.set_xticks(block_sizes)

    # Annotate cache footprint at key sizes.
    ax.text(96, ax.get_ylim()[0], "  96x8=768B", fontsize=7, color="gray", va="bottom")
    ax.text(128, ax.get_ylim()[0], "  128x8=1KB", fontsize=7, color="gray", va="bottom")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35, linestyle="--")
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    cmap = plt.get_cmap("viridis")
    colors = cmap([i / max(len(thread_values) - 1, 1) for i in range(len(thread_values))])

    for color, threads in zip(colors, thread_values):
        curve = agg[agg["threads"] == threads].sort_values("BLOCK_NB")
        best = summary[summary["threads"] == threads].iloc[0]
        ax1.errorbar(
            curve["BLOCK_NB"], curve["gflops_mean"],
            yerr=curve["gflops_std"],
            marker="o", linewidth=1.6, capsize=3,
            color=color, label=f"{threads} thread{'s' if threads != 1 else ''}",
        )
        ax1.scatter(
            [int(best["best_block_nb"])],
            [float(best["best_gflops_mean"])],
            color=color, s=28, zorder=4,
        )

    ax1.set_xlabel("Panel width BLOCK_NB (doubles)")
    ax1.set_ylabel("Performance (GFLOP/s)")
    ax1.set_xticks(block_sizes)
    ax1.set_title("GFLOP/s vs panel width")
    ax1.grid(True, alpha=0.35, linestyle="--")
    ax1.legend(title="Threads", fontsize=8)

    ax2.plot(
        summary["threads"], summary["best_block_nb"],
        marker="o", linewidth=1.6, color="#d62728",
    )
    for _, row in summary.iterrows():
        ax2.annotate(
            f"{row['best_gflops_mean']:.1f} GF/s",
            (row["threads"], row["best_block_nb"]),
            textcoords="offset points", xytext=(0, 6),
            ha="center", fontsize=7,
        )

    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Best BLOCK_NB")
    ax2.set_title("Best panel width by thread count")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks(thread_values)
    ax2.set_yticks(block_sizes)
    ax2.grid(True, alpha=0.35, linestyle="--")

    fig.suptitle(
        f"Panel-width sweep across thread counts\n"
        f"{version}, n={n}, CSD3 icelake, {reps} reps per (NB, thread) point",
        fontsize=10,
    )

fig.tight_layout()
fig.savefig(OUT_PDF)
print(f"Saved {OUT_PDF}")
print(f"Saved {SUMMARY_CSV}")
print("\nBest BLOCK_NB by thread count:")
print(summary.to_string(index=False))
