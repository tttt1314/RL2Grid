#!/usr/bin/env python3
"""Standalone evaluator for saved RL2Grid checkpoints on a chosen chronics split (E3/RD7)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch as th

RL2GRID_ROOT = Path(__file__).resolve().parents[1]
if str(RL2GRID_ROOT) not in sys.path:
    sys.path.insert(0, str(RL2GRID_ROOT))


def _build_model(alg: str, dummy_envs, run_args, ckpt: dict):
    continuous = (run_args.action_type == "redispatch")
    if alg == "PPO":
        from alg.ppo.agent import Agent
        model = Agent(dummy_envs, run_args, continuous)
        model.actor.load_state_dict(ckpt["actor"])
        model.critic.load_state_dict(ckpt["critic"])
    elif alg == "LAGRPPO":
        from alg.lagr_ppo.agent import Agent
        model = Agent(dummy_envs, run_args, continuous)
        model.actor.load_state_dict(ckpt["actor"])
        model.critic.load_state_dict(ckpt["critic"])
        model.cost_critic.load_state_dict(ckpt["cost_critic"])
    elif alg == "DQN":
        from alg.dqn.agent import QNetwork
        model = QNetwork(dummy_envs, run_args)
        model.load_state_dict(ckpt["qnet"])
    elif alg == "SAC":
        from alg.sac.agent import Actor
        model = Actor(dummy_envs, run_args, continuous)
        model.load_state_dict(ckpt["actor"])
    elif alg == "TD3":
        from alg.td3.agent import Actor
        model = Actor(dummy_envs, run_args)
        model.load_state_dict(ckpt["actor"])
    else:
        raise ValueError(f"Unsupported algorithm in checkpoint: {alg}")
    model.eval()
    return model


def _warmup_obs_rms(run_args, model, warmup_steps: int, warmup_split: str, seed: int):
    from env.utils import auxiliary_make_env

    w_args = deepcopy(run_args)
    w_args.chronics_split = warmup_split
    w_args.eval_chronics_split = warmup_split
    w_args.seed = seed
    w_env, _ = auxiliary_make_env(w_args, eval_env=False, idx=0)
    try:
        obs, _ = w_env.reset()
        for _ in range(warmup_steps):
            with th.no_grad():
                act = model.get_eval_action(th.tensor(obs, dtype=th.float32)).detach().cpu().numpy()
            obs, _, terminated, truncated, _ = w_env.step(act)
            if terminated or truncated:
                obs, _ = w_env.reset()
        return deepcopy(w_env.obs_rms)
    finally:
        w_env.close()


def compute_safe_metrics(survs: np.ndarray, raw_overloads: np.ndarray, max_steps: float) -> dict[str, float]:
    """Compute survival, full-horizon compliance, tail-risk (p10/CVaR10), and raw violation-rate metrics."""
    n = int(len(survs))
    mean_s = float(np.mean(survs)) if n > 0 else 0.0
    std_s = float(np.std(survs, ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std_s / math.sqrt(n) if n > 0 else 0.0
    median_s = float(np.median(survs)) if n > 0 else 0.0
    p10_s = float(np.percentile(survs, 10)) if n > 0 else 0.0
    k10 = max(1, int(math.ceil(0.1 * n))) if n > 0 else 0
    cvar10 = float(np.mean(np.sort(survs)[:k10])) if n > 0 else 0.0
    full_rate = float(np.mean(survs >= 1.0 - 1e-6)) if n > 0 else 0.0
    fail_rate = 1.0 - full_rate
    mean_raw_cost = float(np.mean(raw_overloads)) if len(raw_overloads) > 0 else 0.0
    max_raw_cost = float(np.max(raw_overloads)) if len(raw_overloads) > 0 else 0.0
    total_steps = float(np.sum(survs * max_steps))
    viol_per_1k = float(1000.0 * np.sum(raw_overloads) / max(1.0, total_steps))
    min_s = float(np.min(survs)) if n > 0 else 0.0
    max_s = float(np.max(survs)) if n > 0 else 0.0
    return {
        "n": n,
        "mean_s": mean_s,
        "std_s": std_s,
        "ci95": ci95,
        "median_s": median_s,
        "p10_s": p10_s,
        "cvar10": cvar10,
        "full_rate": full_rate,
        "fail_rate": fail_rate,
        "mean_raw_cost": mean_raw_cost,
        "max_raw_cost": max_raw_cost,
        "total_steps": total_steps,
        "viol_per_1k": viol_per_1k,
        "min_s": min_s,
        "max_s": max_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved RL2Grid checkpoint on any chronics split.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint .tar file")
    parser.add_argument("--chronics-split", type=str, default="test", choices=["all", "train", "val", "test"])
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-workers", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=5000,
                        help="Warmup steps to rebuild obs_rms if checkpoint lacks saved obs_rms")
    parser.add_argument("--warmup-split", type=str, default="train", choices=["all", "train", "val", "test"])
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional path to write full JSON evaluation metrics")
    cli = parser.parse_args()

    from env.eval import EvalPool
    from env.utils import auxiliary_make_env

    t0 = time.time()
    ckpt = th.load(cli.ckpt, map_location="cpu", weights_only=False)
    run_args = ckpt["args"]
    alg = run_args.alg.upper()

    eval_args = deepcopy(run_args)
    eval_args.seed = cli.seed
    eval_args.eval_episodes = cli.eval_episodes
    eval_args.eval_workers = cli.eval_workers
    eval_args.chronics_split = cli.chronics_split
    eval_args.eval_chronics_split = cli.chronics_split
    # Always measure raw overload + disconnection violations uniformly across all algorithms.
    eval_args.constraints_type = 2
    eval_args.fail_absorbing_cost = 0.0
    eval_args.keep_disc_penalty = True

    sample_env, _ = auxiliary_make_env(eval_args, eval_env=True, idx=0)
    max_steps = float(sample_env.init_env.chronics_handler.max_episode_duration())
    dummy_envs = argparse.Namespace(
        single_observation_space=sample_env.observation_space,
        single_action_space=sample_env.action_space,
    )
    sample_env.close()

    model = _build_model(alg, dummy_envs, run_args, ckpt)

    obs_rms = None
    rms_src = "disabled"
    if getattr(run_args, "norm_obs", False):
        if "obs_rms" in ckpt:
            obs_rms = deepcopy(ckpt["obs_rms"])
            rms_src = "checkpoint"
        else:
            obs_rms = _warmup_obs_rms(run_args, model, cli.warmup_steps, cli.warmup_split, cli.seed + 500)
            rms_src = f"warmup({cli.warmup_steps}@{cli.warmup_split})"

    n_tags = 3 + (1 if run_args.action_type == "topology" else 0) + (1 if getattr(run_args, "n1_reward", False) else 0)
    n_workers = min(cli.eval_workers, cli.eval_episodes)

    pool = EvalPool(eval_args, n_workers=n_workers, n_tags=n_tags, base_idx=cli.seed)
    try:
        results = pool.run(model, obs_rms, cli.eval_episodes, cmdp=True)
    finally:
        pool.close()

    survs = np.array([r[0] for r in results], dtype=float)
    raw_overloads = np.array([r[2] for r in results], dtype=float)
    metrics = compute_safe_metrics(survs, raw_overloads, max_steps)
    dt = time.time() - t0
    print(
        f"CKPT={cli.ckpt.name} | alg={alg} | split={cli.chronics_split} | obs_rms={rms_src} | "
        f"n={metrics['n']} | survival={metrics['mean_s']:.4f} ± {metrics['ci95']:.4f} "
        f"(std={metrics['std_s']:.4f}, median={metrics['median_s']:.4f}, p10={metrics['p10_s']:.4f}, "
        f"cvar10={metrics['cvar10']:.4f}, full_rate={metrics['full_rate']:.3f}) | "
        f"raw_overload={metrics['mean_raw_cost']:.2f} (max={metrics['max_raw_cost']:.1f}) | "
        f"viol_per_1k={metrics['viol_per_1k']:.3f} | time={dt:.1f}s"
    )

    if cli.json_out is not None:
        cli.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ckpt": str(cli.ckpt),
            "alg": alg,
            "split": cli.chronics_split,
            "obs_rms_source": rms_src,
            "seed": cli.seed,
            "max_steps": max_steps,
            "metrics": metrics,
            "survs": survs.tolist(),
            "raw_overloads": raw_overloads.tolist(),
        }
        cli.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
