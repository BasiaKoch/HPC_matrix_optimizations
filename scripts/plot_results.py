#!/usr/bin/env python3
"""
plot_results.py — Generate all performance figures for the C2 HPC coursework report.

Usage:
    python3 scripts/plot_results.py

Outputs (saved to report/figures/):
    fig1_serial_gflops.pdf       — Serial GFLOPS comparison (v1, v2, v3)
    fig2_scaling_gflops.pdf      — Strong scaling: GFLOPS vs threads
    fig3_scaling_speedup.pdf     — Strong scaling: speedup vs threads
    fig4_scaling_efficiency.pdf  — Strong scaling: parallel efficiency vs threads
    fig5_problem_size.pdf        — GFLOPS vs n at selected thread counts (v5)
    fig6_v3_vs_v5_n8000.pdf      — v3 vs v5 head-to-head at n=8000
    fig7_serial_time.pdf         — Wall-clock time vs n (serial versions)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; works on CSD3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIAL_CSV  = os.path.join(ROOT, "results", "csd3_serial.csv")
SCALING_CSV = os.path.join(ROOT, "results", "csd3_scaling.csv")
FIG_DIR     = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Matplotlib style
# ------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "lines.linewidth":   1.6,
    "lines.markersize":  5,
    "errorbar.capsize":  3,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

COLORS = {
    "v1_baseline":       "#d62728",
    "v2_serial_opt":     "#1f77b4",
    "v3_serial_opt":     "#2ca02c",
    "v3_openmp":         "#ff7f0e",
    "v5_blocked_NB128":  "#9467bd",
}
MARKERS = {
    "v3_openmp":         "o",
    "v5_blocked_NB128":  "s",
}
LABELS = {
    "v1_baseline":      "v1 baseline (−O0)",
    "v2_serial_opt":    "v2 serial opt (−O3)",
    "v3_serial_opt":    "v3 serial opt (−O3)",
    "v3_openmp":        "v3 OpenMP (flat parallel)",
    "v5_blocked_NB128": "v5 panel-blocked (NB=128)",
}

# ------------------------------------------------------------------
# Helper: reduce reps → mean ± std
# ------------------------------------------------------------------
def reduce(df, group_cols):
    agg = df.groupby(group_cols).agg(
        time_mean=("time_s", "mean"),
        time_std =("time_s", "std"),
        gflops_mean=("gflops", "mean"),
        gflops_std =("gflops", "std"),
    ).reset_index()
    return agg

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
serial  = pd.read_csv(SERIAL_CSV)
scaling = pd.read_csv(SCALING_CSV)

serial_agg  = reduce(serial,  ["version", "n", "threads"])
scaling_agg = reduce(scaling, ["version", "n", "threads"])

# ==================================================================
# Fig 1 — Serial GFLOPS comparison
# ==================================================================
def fig1():
    vers  = ["v1_baseline", "v2_serial_opt", "v3_serial_opt"]
    ns    = sorted(serial_agg["n"].unique())
    x     = np.arange(len(ns))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, v in enumerate(vers):
        d = serial_agg[serial_agg["version"] == v].sort_values("n")
        # Align to full n-axis (missing sizes get height=0)
        heights = []
        errs    = []
        xpos    = []
        for idx, n in enumerate(ns):
            row = d[d["n"] == n]
            if len(row):
                heights.append(float(row["gflops_mean"].iloc[0]))
                errs.append(float(row["gflops_std"].iloc[0]))
                xpos.append(x[idx] + (k - 1) * width)
        ax.bar(xpos, heights, width,
               label=LABELS[v], color=COLORS[v],
               yerr=errs, capsize=3, error_kw={"elinewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Fig 1 — Serial optimisation: GFLOP/s comparison\n"
                 "(CSD3 icelake, 1 thread, 3 reps, error bars = ±1 SD)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_serial_gflops.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ==================================================================
# Fig 2 — Strong scaling: GFLOPS vs threads (one subplot per n)
# ==================================================================
def fig2():
    ns   = sorted(scaling_agg["n"].unique())
    vers = ["v3_openmp", "v5_blocked_NB128"]
    fig, axes = plt.subplots(1, len(ns), figsize=(13, 4), sharey=False)

    for ax, n in zip(axes, ns):
        for v in vers:
            d = scaling_agg[(scaling_agg["version"] == v) &
                            (scaling_agg["n"] == n)].sort_values("threads")
            ax.errorbar(d["threads"], d["gflops_mean"], yerr=d["gflops_std"],
                        label=LABELS[v], color=COLORS[v],
                        marker=MARKERS[v], linewidth=1.6)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Threads")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([1, 2, 4, 8, 16, 32, 48, 64, 76])
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Performance (GFLOP/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.suptitle("Fig 2 — Strong scaling: GFLOP/s vs thread count\n"
                 "(CSD3 icelake, v3_openmp vs v5_blocked_NB128, 3 reps, error bars = ±1 SD)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_scaling_gflops.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ==================================================================
# Fig 3 — Strong scaling: speedup vs threads
# ==================================================================
def fig3():
    ns   = sorted(scaling_agg["n"].unique())
    vers = ["v3_openmp", "v5_blocked_NB128"]
    fig, axes = plt.subplots(1, len(ns), figsize=(13, 4), sharey=False)

    for ax, n in zip(axes, ns):
        # ideal speedup line
        tmax = scaling_agg["threads"].max()
        t_range = np.array([1, 2, 4, 8, 16, 32, 48, 64, 76])
        ax.plot(t_range, t_range, "k--", linewidth=0.9, label="Ideal (linear)")

        for v in vers:
            d = scaling_agg[(scaling_agg["version"] == v) &
                            (scaling_agg["n"] == n)].sort_values("threads")
            t1 = float(d[d["threads"] == 1]["time_mean"].iloc[0])
            speedup = t1 / d["time_mean"].values
            threads = d["threads"].values
            ax.plot(threads, speedup, label=LABELS[v],
                    color=COLORS[v], marker=MARKERS[v], linewidth=1.6)

        ax.set_title(f"n = {n}")
        ax.set_xlabel("Threads")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([1, 2, 4, 8, 16, 32, 48, 64, 76])
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Speedup (T₁ / Tₚ)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.suptitle("Fig 3 — Strong scaling: speedup vs thread count\n"
                 "(CSD3 icelake, v3_openmp vs v5_blocked_NB128)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_scaling_speedup.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ==================================================================
# Fig 4 — Strong scaling: parallel efficiency vs threads
# ==================================================================
def fig4():
    ns   = sorted(scaling_agg["n"].unique())
    vers = ["v3_openmp", "v5_blocked_NB128"]
    fig, axes = plt.subplots(1, len(ns), figsize=(13, 4), sharey=True)

    for ax, n in zip(axes, ns):
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.9, label="Ideal (100%)")
        for v in vers:
            d = scaling_agg[(scaling_agg["version"] == v) &
                            (scaling_agg["n"] == n)].sort_values("threads")
            t1 = float(d[d["threads"] == 1]["time_mean"].iloc[0])
            eff = t1 / (d["time_mean"].values * d["threads"].values)
            ax.plot(d["threads"].values, eff,
                    label=LABELS[v], color=COLORS[v],
                    marker=MARKERS[v], linewidth=1.6)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Threads")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([1, 2, 4, 8, 16, 32, 48, 64, 76])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.25)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    axes[0].set_ylabel("Parallel efficiency (speedup / threads)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Fig 4 — Strong scaling: parallel efficiency vs thread count\n"
                 "(CSD3 icelake, v3_openmp vs v5_blocked_NB128)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_scaling_efficiency.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# ==================================================================
# Fig 5 — GFLOPS vs problem size n (v5 only, selected thread counts)
# ==================================================================
def fig5():
    v = "v5_blocked_NB128"
    selected_threads = [1, 8, 32, 76]
    thread_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for t, col in zip(selected_threads, thread_colors):
        d = scaling_agg[(scaling_agg["version"] == v) &
                        (scaling_agg["threads"] == t)].sort_values("n")
        ax.errorbar(d["n"], d["gflops_mean"], yerr=d["gflops_std"],
                    label=f"{t} thread{'s' if t > 1 else ''}",
                    color=col, marker="o", linewidth=1.6)

    ax.set_xlabel("Matrix dimension n")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Fig 5 — GFLOP/s vs problem size (v5 panel-blocked, NB=128)\n"
                 "(CSD3 icelake, 3 reps, error bars = ±1 SD)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_problem_size.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ==================================================================
# Fig 6 — v3 vs v5 head-to-head at n=8000
# ==================================================================
def fig6():
    n    = 8000
    vers = ["v3_openmp", "v5_blocked_NB128"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for v in vers:
        d = scaling_agg[(scaling_agg["version"] == v) &
                        (scaling_agg["n"] == n)].sort_values("threads")
        ax1.errorbar(d["threads"], d["gflops_mean"], yerr=d["gflops_std"],
                     label=LABELS[v], color=COLORS[v],
                     marker=MARKERS[v], linewidth=1.6)
        t1  = float(d[d["threads"] == 1]["time_mean"].iloc[0])
        eff = t1 / (d["time_mean"].values * d["threads"].values)
        ax2.plot(d["threads"].values, eff,
                 label=LABELS[v], color=COLORS[v],
                 marker=MARKERS[v], linewidth=1.6)

    ax1.axhline(scaling_agg[(scaling_agg["version"] == "v5_blocked_NB128") &
                             (scaling_agg["n"] == n) &
                             (scaling_agg["threads"] == 76)]["gflops_mean"].values[0],
               color=COLORS["v5_blocked_NB128"], linestyle=":", linewidth=0.8)
    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Performance (GFLOP/s)")
    ax1.set_title(f"GFLOP/s  (n={n})")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 48, 64, 76])
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()

    ax2.axhline(1.0, color="k", linestyle="--", linewidth=0.9, label="Ideal")
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Parallel efficiency")
    ax2.set_title(f"Efficiency  (n={n})")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 48, 64, 76])
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_ylim(0, 1.25)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.legend()

    fig.suptitle(f"Fig 6 — v3 (flat parallel) vs v5 (panel-blocked) at n={n}\n"
                 "(CSD3 icelake, 3 reps)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_v3_vs_v5_n8000.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ==================================================================
# Fig 7 — Wall-clock time vs n (serial comparison)
# ==================================================================
def fig7():
    vers = ["v1_baseline", "v2_serial_opt", "v3_serial_opt"]
    fig, ax = plt.subplots(figsize=(6, 4))

    for v in vers:
        d = serial_agg[serial_agg["version"] == v].sort_values("n")
        ax.errorbar(d["n"], d["time_mean"], yerr=d["time_std"],
                    label=LABELS[v], color=COLORS[v],
                    marker="o", linewidth=1.6)

    # O(n^3) reference line anchored at v2, n=2000
    ref_v = serial_agg[(serial_agg["version"] == "v2_serial_opt") &
                       (serial_agg["n"] == 2000)]
    t_ref = float(ref_v["time_mean"].iloc[0])
    n_ref = 2000
    ns_plot = np.logspace(np.log10(400), np.log10(7000), 100)
    ax.plot(ns_plot, t_ref * (ns_plot / n_ref)**3,
            "k--", linewidth=0.9, label=r"$O(n^3)$ reference")

    ax.set_xlabel("Matrix dimension n")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fig 7 — Serial wall-clock time vs n\n"
                 "(CSD3 icelake, 1 thread, 3 reps, error bars = ±1 SD)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig7_serial_time.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    print(f"Reading serial data from  : {SERIAL_CSV}")
    print(f"Reading scaling data from : {SCALING_CSV}")
    print(f"Saving figures to         : {FIG_DIR}\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("\nAll figures generated successfully.")
