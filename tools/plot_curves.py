#!/usr/bin/env python3
"""Generate seed-0 baseline learning curves and E1/E2/E1B diagnostic plots from VM logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

EVAL_RE = re.compile(
    r"Eval at step (\d+), survival=([\d.]+)%(?:, return=\[(.*?)\])?(?:, cost_return=\[([\d.eE+-]+)\])?"
)
ROLL_RE = re.compile(
    r"LagrRollout iter=(\d+) step=(\d+) J_c=([\d.eE+-]+) thresh=([\d.eE+-]+) "
    r"lag_before=([\d.eE+-]+) lag=([\d.eE+-]+) nu=([\d.eE+-]+) "
    r"n_ep=(\d+) ep_cost=([\w.eE+-]+) ep_len=([\w.eE+-]+) "
    r"kl=([\d.eE+-]+) clipfrac=([\d.eE+-]+) "
    r"pg_r=([\d.eE+-]+) surr_c=([\d.eE+-]+) ent=([\d.eE+-]+)"
    r"(?: ep_raw_cost=([\w.eE+-]+) ep_rate=([\w.eE+-]+) epochs=(\d+))?"
)


def parse_job(job_dir: Path) -> dict:
    eval_map: dict[int, tuple[float, float | None]] = {}
    rollouts: list[dict[str, float | int | None]] = []
    for log_path in sorted(job_dir.glob("seg_*.log")):
        for line in log_path.read_text(errors="ignore").splitlines():
            m = EVAL_RE.search(line)
            if m:
                step = int(m.group(1))
                surv = float(m.group(2)) / 100.0
                cost_ret = float(m.group(4)) if m.group(4) is not None else None
                eval_map[step] = (surv, cost_ret)
                continue
            mr = ROLL_RE.search(line)
            if mr:
                rollouts.append({
                    "iter": int(mr.group(1)),
                    "step": int(mr.group(2)),
                    "J_c": float(mr.group(3)),
                    "thresh": float(mr.group(4)),
                    "lag_before": float(mr.group(5)),
                    "lag": float(mr.group(6)),
                    "nu": float(mr.group(7)),
                    "n_ep": int(mr.group(8)),
                    "ep_cost": float(mr.group(9)),
                    "ep_len": float(mr.group(10)),
                    "kl": float(mr.group(11)),
                    "clipfrac": float(mr.group(12)),
                    "pg_r": float(mr.group(13)),
                    "surr_c": float(mr.group(14)),
                    "ent": float(mr.group(15)),
                    "ep_raw_cost": float(mr.group(16)) if mr.group(16) is not None else None,
                    "ep_rate": float(mr.group(17)) if mr.group(17) is not None else None,
                    "epochs": int(mr.group(18)) if mr.group(18) is not None else None,
                })
    steps = np.array(sorted(eval_map), dtype=float)
    survs = np.array([eval_map[int(s)][0] for s in steps], dtype=float)
    costs = np.array(
        [eval_map[int(s)][1] if eval_map[int(s)][1] is not None else np.nan for s in steps],
        dtype=float,
    )
    return {"steps": steps, "survs": survs, "costs": costs, "rollouts": rollouts}


def trailing_ma(vals: np.ndarray, window: int = 3) -> np.ndarray:
    out = np.empty_like(vals, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out[i] = float(np.mean(vals[lo : i + 1]))
    return out


def step_formatter(x: float, _pos: int) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1_000_000:
        v = x / 1_000_000.0
        return f"{v:.1f}M" if abs(v - round(v)) > 1e-6 else f"{int(round(v))}M"
    return f"{int(round(x / 1000.0))}k"


def style_axes(ax: plt.Axes, max_step: float | None = None, ylim: tuple[float, float] | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
    if max_step is not None:
        ax.set_xlim(0, max_step * 1.02)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_eval_series(
    ax: plt.Axes,
    data: dict,
    label: str,
    color: str,
    linestyle: str = "-",
    max_step: float | None = None,
    linewidth: float = 2.0,
) -> None:
    steps, survs = data["steps"], data["survs"]
    if max_step is not None:
        mask = steps <= max_step
        steps, survs = steps[mask], survs[mask]
    ax.plot(steps, survs, color=color, linestyle=linestyle, alpha=0.22, linewidth=1.0)
    ax.plot(steps, trailing_ma(survs, 3), color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def make_baseline_figure(jobs: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    fig.suptitle("RL2Grid bus14 (difficulty 0, seed 0) — Baseline Learning Curves", fontsize=13.5, fontweight="bold", y=0.98)

    # Panel A: Topology Vanilla (0-5M)
    ax = axes[0, 0]
    plot_eval_series(ax, jobs["PPO_T_V_s0"], "PPO V (last3=0.67, best=0.86)", "tab:blue")
    plot_eval_series(ax, jobs["DQN_T_V_s0"], "DQN V (last3=0.10, best=0.34)", "tab:orange")
    plot_eval_series(ax, jobs["SAC_T_V_s0"], "SAC V (last3=0.01, best=0.35@250k)", "tab:green")
    ax.axhline(0.74, color="tab:blue", linestyle=":", linewidth=1.0, alpha=0.55, label="Paper Table 2 (PPO=0.74 / SAC=0.17 / DQN=0.07)")
    ax.axhline(0.17, color="tab:green", linestyle=":", linewidth=1.0, alpha=0.55)
    ax.axhline(0.07, color="tab:orange", linestyle=":", linewidth=1.0, alpha=0.55)
    style_axes(ax, max_step=5_060_000, ylim=(-0.02, 1.03))
    ax.set_title("A. Topology Vanilla (0–5M steps)")
    ax.set_xlabel("Environment / Agent Steps")
    ax.set_ylabel("Eval Survival (10 eps, 3-eval MA)")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, framealpha=0.9)

    # Panel B: Topology Heuristics I & R (0-500k)
    ax = axes[0, 1]
    plot_eval_series(ax, jobs["PPO_T_I_s0"], "PPO I (last3=0.90)", "tab:blue", "-")
    plot_eval_series(ax, jobs["PPO_T_R_s0"], "PPO R (last3=0.80)", "tab:blue", "--")
    plot_eval_series(ax, jobs["DQN_T_I_s0"], "DQN I (last3=0.97)", "tab:orange", "-")
    plot_eval_series(ax, jobs["DQN_T_R_s0"], "DQN R (last3=0.68)", "tab:orange", "--")
    plot_eval_series(ax, jobs["SAC_T_I_s0"], "SAC I (last3=0.85)", "tab:green", "-")
    plot_eval_series(ax, jobs["SAC_T_R_s0"], "SAC R (last3=0.62)", "tab:green", "--")
    style_axes(ax, max_step=500_000, ylim=(-0.02, 1.06))
    ax.set_title("B. Topology Heuristics — Idle (solid) vs. Recovery (dashed)")
    ax.set_xlabel("Agent Decision Steps (0–500k)")
    ax.set_ylabel("Eval Survival (10 eps, 3-eval MA)")
    ax.legend(loc="lower right", ncol=2, fontsize=8.0, frameon=True, framealpha=0.9)

    # Panel C: Redispatch Vanilla (0-5M)
    ax = axes[1, 0]
    plot_eval_series(ax, jobs["PPO_R_s0"], "PPO R (last3=0.16, best=0.40)", "tab:blue")
    plot_eval_series(ax, jobs["SAC_R_s0"], "SAC R (last3=0.19, best=0.42)", "tab:green")
    plot_eval_series(ax, jobs["TD3_R_s0"], "TD3 R (last3=0.15, best=0.37)", "tab:red")
    ax.axhline(0.17, color="tab:blue", linestyle=":", linewidth=1.0, alpha=0.55, label="Paper Table 3 (PPO=0.17 / TD3=0.06 / SAC=0.01)")
    ax.axhline(0.06, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.55)
    ax.axhline(0.01, color="tab:green", linestyle=":", linewidth=1.0, alpha=0.55)
    style_axes(ax, max_step=5_000_000, ylim=(-0.02, 0.54))
    ax.set_title("C. Redispatch Vanilla (0–5M steps, with PR #7 action clip)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Survival (10 eps, 3-eval MA)")
    ax.legend(loc="upper right", fontsize=8.0, frameon=True, framealpha=0.9)

    # Panel D: Constrained LagrPPO vs PPO_T_V (0-3M)
    ax = axes[1, 1]
    plot_eval_series(ax, jobs["PPO_T_V_s0"], "PPO V @<=3M (last3=0.71, best=0.84)", "0.35", "--", max_step=3_000_000)
    plot_eval_series(ax, jobs["LAGRPPO_C1_s0"], "LagrPPO C1 LSI, d=50 (last3=0.32, best=0.51)", "tab:red", "-")
    plot_eval_series(ax, jobs["LAGRPPO_C2_s0"], "LagrPPO C2 TLO, d=0 (last3=0.50, best=0.79)", "tab:purple", "-")
    style_axes(ax, max_step=3_000_000, ylim=(-0.02, 1.03))
    ax.set_title("D. Constrained Topology (LagrPPO, 0–3M) vs. PPO V")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Survival (10 eps, 3-eval MA)")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_diagnostic_figure(jobs: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    fig.suptitle("LagrPPO Diagnostic Experiments (E1, E2, E1B) — bus14, seed 0", fontsize=13.5, fontweight="bold", y=0.98)

    # Panel A: Eval Survival (0-3M)
    ax = axes[0, 0]
    plot_eval_series(ax, jobs["PPO_T_V_s0"], "PPO V @<=3M (0.710)", "0.40", "--", max_step=3_000_000, linewidth=1.7)
    plot_eval_series(ax, jobs["E1_LAGRPPO_C1_LAM0_s0"], "E1: C1 hyperparams, nu~0 (0.579, best 0.643)", "tab:green", "-")
    plot_eval_series(ax, jobs["E2_LAGRPPO_C1_s0"], "E2-C1: LSI thresh=50 (0.355)", "tab:red", "-")
    plot_eval_series(ax, jobs["E2_LAGRPPO_C2_s0"], "E2-C2: TLO thresh=0 (0.559, best 0.812)", "tab:purple", "-")
    plot_eval_series(ax, jobs["E1B_LAGRPPO_C1_PPOCFG_s0"], "E1B: nu~0 + PPO cfg 3M (0.250)", "tab:orange", "-")
    style_axes(ax, max_step=3_000_000, ylim=(-0.02, 1.03))
    ax.set_title("A. Eval Survival: Disabling nu Recovers Learning (E1 vs. E2-C1)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Survival (10 eps, 3-eval MA)")
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.9)

    # Panel B: E2-C1 Dual Dynamics (O2/O4)
    ax = axes[0, 1]
    r_c1 = jobs["E2_LAGRPPO_C1_s0"]["rollouts"]
    s_c1 = np.array([r["step"] for r in r_c1], dtype=float)
    jc_c1 = np.array([r["J_c"] for r in r_c1], dtype=float)
    lag_c1 = np.array([r["lag"] for r in r_c1], dtype=float)
    nu_c1 = np.array([r["nu"] for r in r_c1], dtype=float)

    ax.axvspan(0, 1_450_000, color="tab:red", alpha=0.06)
    ax.axvspan(1_450_000, 3_050_000, color="tab:green", alpha=0.06)
    l1, = ax.plot(s_c1, jc_c1, "o-", color="tab:blue", markersize=3.5, linewidth=1.8, label="J_c (last 8k failure sum, left)")
    l2 = ax.axhline(50.0, color="tab:blue", linestyle=":", linewidth=1.3, label="threshold d = 50 (left)")
    ax.set_yscale("log")
    style_axes(ax, max_step=3_000_000, ylim=(8, 3200))
    ax.set_title("B. E2-C1 Dual Dynamics: Early Overshoot & Adam Momentum Lock")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("J_c (log scale)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    l3, = ax2.plot(s_c1, lag_c1, "s-", color="tab:red", markersize=3.2, linewidth=1.8, label="lag_mul (right)")
    l4, = ax2.plot(s_c1, nu_c1, "^--", color="tab:orange", markersize=3.2, linewidth=1.6, label="nu = softplus(lag) (right)")
    ax2.set_ylim(0.72, 1.95)
    ax2.set_ylabel("Dual Multiplier (lag / nu)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax.text(720_000, 2250, "iter 1–14: J_c > 50\n(lambda pushed up)", ha="center", va="top", fontsize=7.8, color="darkred")
    ax.text(2_250_000, 2250, "iter 15–30: J_c < 50\n(Adam inertia keeps rising)", ha="center", va="top", fontsize=7.8, color="darkgreen")
    ax2.annotate(
        "peaks 1.504 @ 2.8–2.9M",
        xy=(2_800_000, 1.504),
        xytext=(1_800_000, 1.28),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0),
        fontsize=7.8,
        color="tab:red",
    )
    lines = [l1, l2, l3, l4]
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=7.6, frameon=True, framealpha=0.92)

    # Panel C: Trust-Region Breakdown approx_kl (O5)
    ax = axes[1, 0]
    for job_id, label, color, ls in [
        ("E1_LAGRPPO_C1_LAM0_s0", "E1 (nu~0): kl <= 0.049", "tab:green", "-"),
        ("E2_LAGRPPO_C1_s0", "E2-C1 (nu=1.35->1.70): iter1 kl=0.707", "tab:red", "-"),
        ("E2_LAGRPPO_C2_s0", "E2-C2 (nu=1.35->1.58): iter1 kl=5.52", "tab:purple", "-"),
        ("E1B_LAGRPPO_C1_PPOCFG_s0", "E1B (PPO cfg 3M): lr decays -> 0", "tab:orange", "-"),
    ]:
        rs = jobs[job_id]["rollouts"]
        xs = np.array([r["step"] for r in rs], dtype=float)
        ys = np.maximum(np.array([r["kl"] for r in rs], dtype=float), 1e-5)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.8 if job_id != "E1B_LAGRPPO_C1_PPOCFG_s0" else 1.4, label=label)
    ax.axhline(0.02, color="0.4", linestyle=":", linewidth=1.2, label="KL = 0.02 reference")
    ax.set_yscale("log")
    style_axes(ax, max_step=3_000_000, ylim=(5e-6, 15))
    ax.set_title("C. Trust-Region Breakdown: Unclipped Cost Surrogate (approx_kl)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Approx KL (log scale)")
    ax.legend(loc="lower left", fontsize=7.8, frameon=True, framealpha=0.92)

    # Panel D: E2-C2 Oscillation & Late Cost Surge
    ax = axes[1, 1]
    e2c2 = jobs["E2_LAGRPPO_C2_s0"]
    steps_c2, survs_c2, costs_c2 = e2c2["steps"], e2c2["survs"], e2c2["costs"]
    r_c2 = e2c2["rollouts"]
    s_r2 = np.array([r["step"] for r in r_c2], dtype=float)
    lag_r2 = np.array([r["lag"] for r in r_c2], dtype=float)

    b1, = ax.plot(steps_c2, costs_c2, "o-", color="tab:red", markersize=3.5, linewidth=1.7, label="eval_cost (left)")
    style_axes(ax, max_step=3_000_000, ylim=(0, 145))
    ax.set_title("D. E2-C2 (thresh=0): Mid-Run Collapse & Late Eval Cost Surge")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Cost Return (disconnections + overloads)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    b2, = ax_r.plot(steps_c2, survs_c2, "s--", color="tab:purple", markersize=3.2, linewidth=1.6, label="eval_surv (right)")
    b3, = ax_r.plot(s_r2, lag_r2 - 1.0, ":", color="tab:blue", linewidth=1.5, label="lag_mul - 1.0 (right, 0.05->0.36)")
    ax_r.set_ylim(-0.02, 1.02)
    ax_r.set_ylabel("Eval Survival / (lag_mul - 1.0)", color="tab:purple")
    ax_r.tick_params(axis="y", labelcolor="tab:purple")

    ax_r.annotate(
        "collapse\n(0.105 @ 1.7M)",
        xy=(1_700_000, 0.105),
        xytext=(1_180_000, 0.68),
        arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.0),
        fontsize=7.6,
        color="tab:purple",
    )
    ax.annotate(
        "cost surge\n(126.5 / 119.5)",
        xy=(2_700_000, 126.5),
        xytext=(1_980_000, 118),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0),
        fontsize=7.6,
        color="tab:red",
    )
    lines_d = [b1, b2, b3]
    ax.legend(lines_d, [l.get_label() for l in lines_d], loc="upper left", fontsize=7.8, frameon=True, framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _eval_summary(data: dict, max_step: float | None = None) -> tuple[float, float, float]:
    """Return (last3_survival_mean, best_survival, last3_cost_mean) within max_step."""
    steps, survs, costs = data["steps"], data["survs"], data["costs"]
    if max_step is not None:
        mask = steps <= max_step
        survs = survs[mask]
        costs = costs[mask]
    last3_surv = float(np.mean(survs[-3:]))
    best_surv = float(np.max(survs))
    valid_costs = costs[-3:][~np.isnan(costs[-3:])]
    last3_cost = float(np.mean(valid_costs)) if len(valid_costs) > 0 else float("nan")
    return last3_surv, best_surv, last3_cost


def make_batch1_figure(jobs: dict[str, dict], out_path: Path) -> None:
    """Plot 2x2 comparison across E2 baselines, E6a O5-fix ablation, and E6b PID+CMDP."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=180)
    fig.suptitle(
        "LagrPPO Batch 1 (E6a O5-Fix & E6b PID+CMDP) vs. E1/E2 — bus14, seed 0",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    ppo_s3, ppo_best, _ = _eval_summary(jobs["PPO_T_V_s0"], max_step=3_000_000)

    # Panel A: C1 Failure-Constrained Eval Survival (0-3M)
    ax = axes[0, 0]
    plot_eval_series(
        ax, jobs["PPO_T_V_s0"],
        f"PPO V <=3M (last3={ppo_s3:.3f}, best={ppo_best:.3f})",
        "0.40", "--", max_step=3_000_000, linewidth=1.7,
    )
    c1_specs = [
        ("E1_LAGRPPO_C1_LAM0_s0", "E1: nu~0", "tab:green", "-"),
        ("E2_LAGRPPO_C1_s0", "E2-C1: orig C1", "tab:red", "-"),
        ("E6A_C1_O5FIX_s0", "E6a-C1: O5 fix", "tab:orange", "-"),
        ("E6B_C1_PID_RD1_s0", "E6b-C1: PID+RD1", "tab:blue", "-"),
    ]
    for jid, name, color, ls in c1_specs:
        l3, bst, _ = _eval_summary(jobs[jid])
        plot_eval_series(ax, jobs[jid], f"{name} (last3={l3:.3f}, best={bst:.3f})", color, ls)
    style_axes(ax, max_step=3_000_000, ylim=(-0.02, 1.03))
    ax.set_title("A. Failure-Constrained (C1) Eval Survival")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Survival (10 eps, 3-eval MA)")
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.92)

    # Panel B: C2 Overload-Constrained Eval Survival (0-3M)
    ax = axes[0, 1]
    plot_eval_series(
        ax, jobs["PPO_T_V_s0"],
        f"PPO V <=3M (last3={ppo_s3:.3f})",
        "0.40", "--", max_step=3_000_000, linewidth=1.7,
    )
    c2_specs = [
        ("E2_LAGRPPO_C2_s0", "E2-C2: orig C2", "tab:purple", "-", "c"),
        ("E6A_C2_O5FIX_s0", "E6a-C2: O5 fix", "tab:cyan", "-", "c"),
        ("E6B_C2_PID_RD1_s0", "E6b-C2: PID+RD1", "tab:brown", "-", "c*"),
    ]
    for jid, name, color, ls, c_tag in c2_specs:
        l3, bst, c3 = _eval_summary(jobs[jid])
        plot_eval_series(
            ax, jobs[jid],
            f"{name} (last3={l3:.3f}, best={bst:.3f}, {c_tag}={c3:.1f})",
            color, ls,
        )
    style_axes(ax, max_step=3_000_000, ylim=(-0.02, 1.03))
    ax.set_title("B. Overload-Constrained (C2) Eval Survival (*c incl. absorb)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Survival (10 eps, 3-eval MA)")
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.92)

    # Panel C: Trust-Region approx_kl (log scale)
    ax = axes[1, 0]
    kl_specs = [
        ("E2_LAGRPPO_C1_s0", "E2-C1 (no O5 fix)", "tab:red", "-"),
        ("E2_LAGRPPO_C2_s0", "E2-C2 (no O5 fix)", "tab:purple", "--"),
        ("E6A_C1_O5FIX_s0", "E6a-C1 (O5 fix)", "tab:orange", "-"),
        ("E6A_C2_O5FIX_s0", "E6a-C2 (O5 fix)", "tab:cyan", "-"),
        ("E6B_C1_PID_RD1_s0", "E6b-C1 (PID+RD1)", "tab:blue", "-"),
        ("E6B_C2_PID_RD1_s0", "E6b-C2 (PID+RD1)", "tab:brown", "-"),
    ]
    for jid, name, color, ls in kl_specs:
        rs = jobs[jid]["rollouts"]
        xs = np.array([r["step"] for r in rs], dtype=float)
        ys = np.maximum(np.array([r["kl"] for r in rs], dtype=float), 1e-5)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.7, label=f"{name} (max={float(np.max(ys)):.3f})")
    ax.axhline(0.02, color="0.35", linestyle=":", linewidth=1.3, label="target_kl = 0.02")
    ax.set_yscale("log")
    style_axes(ax, max_step=3_000_000, ylim=(2e-3, 12.0))
    ax.set_title("C. Trust-Region Stability: approx_kl Eliminates Spikes")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Approx KL (log scale)")
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.92)

    # Panel D: Dual Multiplier nu (log scale)
    ax = axes[1, 1]
    nu_specs = [
        ("E2_LAGRPPO_C1_s0", "E2-C1 (Adam, LSI d=50)", "tab:red", "--"),
        ("E2_LAGRPPO_C2_s0", "E2-C2 (Adam, TLO d=0)", "tab:purple", "--"),
        ("E6A_C1_O5FIX_s0", "E6a-C1 (Adam+O5, LSI d=50)", "tab:orange", "-"),
        ("E6A_C2_O5FIX_s0", "E6a-C2 (Adam+O5, TLO d=0)", "tab:cyan", "-"),
        ("E6B_C1_PID_RD1_s0", "E6b-C1 (PID, ep_mean d=0.5)", "tab:blue", "-"),
        ("E6B_C2_PID_RD1_s0", "E6b-C2 (PID, cost_rate d=0.05)", "tab:brown", "-"),
    ]
    for jid, name, color, ls in nu_specs:
        rs = jobs[jid]["rollouts"]
        xs = np.array([r["step"] for r in rs], dtype=float)
        ys = np.array([r["nu"] for r in rs], dtype=float)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.8, label=f"{name} (-> {float(ys[-1]):.2f})")
    ax.set_yscale("log")
    style_axes(ax, max_step=3_000_000, ylim=(0.45, 22.0))
    ax.set_yticks([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_title("D. Dual Weight nu: Adam Lock & PID Integral Windup")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Effective Dual Multiplier nu (log scale)")
    ax.legend(loc="upper left", fontsize=7.2, frameon=True, framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {d.name: parse_job(d) for d in sorted(args.logs_dir.iterdir()) if d.is_dir()}

    make_baseline_figure(jobs, args.out_dir / "bus14_seed0_learning_curves.png")
    make_diagnostic_figure(jobs, args.out_dir / "e1_e2_diagnostic_curves.png")
    batch1_keys = {"E6A_C1_O5FIX_s0", "E6A_C2_O5FIX_s0", "E6B_C1_PID_RD1_s0", "E6B_C2_PID_RD1_s0"}
    if batch1_keys.issubset(jobs):
        make_batch1_figure(jobs, args.out_dir / "e6_batch1_comparison.png")
    print("Generated figures in", args.out_dir)


if __name__ == "__main__":
    main()
