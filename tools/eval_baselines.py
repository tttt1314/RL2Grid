#!/usr/bin/env python3
"""Parallel evaluator for non-RL Grid2Op baselines (Idle, RecoPowerlineAgent, MILP)."""

from __future__ import annotations

import argparse
import inspect
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

RL2GRID_ROOT = Path(__file__).resolve().parents[1]
if str(RL2GRID_ROOT) not in sys.path:
    sys.path.insert(0, str(RL2GRID_ROOT))


def _make_agent(name: str, g2o_env):
    key = name.lower()
    if key == "idle":
        return None
    if key == "reco":
        from grid2op.Agent import RecoPowerlineAgent
        return RecoPowerlineAgent(g2o_env.action_space)
    if key == "milp":
        for mod_name, cls_name in (
            ("grid2op_milp_agent", "MILPAgent"),
            ("l2rpn_baselines.OptimCVXPY", "OptimCVXPY"),
        ):
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                sig = inspect.signature(cls.__init__)
                params = [p for p in sig.parameters.values() if p.name != "self"]
                if len(params) >= 2 and params[1].default is inspect.Parameter.empty:
                    return cls(g2o_env.action_space, g2o_env)
                return cls(g2o_env.action_space)
            except ImportError:
                continue
        raise ImportError(
            "MILP baseline requires grid2op_milp_agent (pip install git+https://github.com/Grid2op/grid2op-milp-agent.git) "
            "or l2rpn_baselines.OptimCVXPY."
        )
    raise ValueError(f"Unknown baseline agent: {name}")


def _overload_and_disc_cost(g2o_env, done: bool) -> float:
    obs = g2o_env.current_obs
    forced = obs.time_next_maintenance == 0
    n_disc = int(np.sum(~obs.line_status & ~forced))
    n_over = 0
    if not done:
        flows = np.abs(g2o_env.backend.get_line_flow())
        limits = np.abs(g2o_env.get_thermal_limit())
        n_over = int(np.sum((limits - flows) < 0))
    return float(n_disc + n_over)


def _worker(args_tuple):
    agent_name, env_args_dict, n_episodes, worker_idx = args_tuple
    import argparse as ap
    from env.utils import auxiliary_make_env

    env_args = ap.Namespace(**env_args_dict)
    gym_env, _ = auxiliary_make_env(env_args, eval_env=True, idx=worker_idx)
    g2o_env = gym_env.init_env
    max_steps = g2o_env.chronics_handler.max_episode_duration()
    agent = _make_agent(agent_name, g2o_env)
    noop = g2o_env.action_space()

    results = []
    try:
        for _ in range(n_episodes):
            gym_env.reset()
            done = False
            ep_c2 = 0.0
            while not done:
                obs = g2o_env.current_obs
                act = noop if agent is None else agent.act(obs, 0.0, False)
                _, _, done, _ = g2o_env.step(act)
                ep_c2 += _overload_and_disc_cost(g2o_env, done)
            surv = float(g2o_env.nb_time_step) / float(max_steps)
            c1_fail = 1.0 if g2o_env.nb_time_step < max_steps else 0.0
            results.append((surv, c1_fail, ep_c2))
    finally:
        gym_env.close()
    return results


def eval_one_agent(agent_name: str, env_args_dict: dict, episodes: int, workers: int, seed: int) -> None:
    t0 = time.time()
    n_workers = min(workers, episodes)
    split = [episodes // n_workers + (1 if i < episodes % n_workers else 0) for i in range(n_workers)]
    tasks = [(agent_name, env_args_dict, k, seed + i) for i, k in enumerate(split) if k > 0]

    try:
        if n_workers == 1:
            rows = _worker(tasks[0])
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_workers) as pool:
                rows = [r for batch in pool.map(_worker, tasks) for r in batch]
    except ImportError as exc:
        print(f"[{agent_name.upper():4s}] SKIPPED: {exc}")
        return

    survs = np.array([r[0] for r in rows], dtype=float)
    fails = np.array([r[1] for r in rows], dtype=float)
    c2s = np.array([r[2] for r in rows], dtype=float)
    mean_s = float(np.mean(survs))
    std_s = float(np.std(survs, ddof=1)) if len(survs) > 1 else 0.0
    ci95 = 1.96 * std_s / math.sqrt(len(survs))
    dt = time.time() - t0
    print(
        f"[{agent_name.upper():4s}] n={len(survs):3d} | survival={mean_s:.4f} ± {ci95:.4f} "
        f"(std={std_s:.4f}, min-max={survs.min():.3f}-{survs.max():.3f}) | "
        f"C1_fail={fails.mean():.3f} | C2_cost={c2s.mean():.2f} | time={dt:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Idle / Reco / MILP baselines on RL2Grid envs.")
    parser.add_argument("--env-id", type=str, default="bus14")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--action-type", type=str, default="topology", choices=["topology", "redispatch"])
    parser.add_argument("--agents", nargs="+", default=["idle", "reco", "milp"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chronics-split", type=str, default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--seed", type=int, default=1000)
    cli, _ = parser.parse_known_args()

    from env.config import get_env_args
    env_args = vars(get_env_args())
    env_args.update({
        "env_id": cli.env_id,
        "difficulty": cli.difficulty,
        "action_type": cli.action_type,
        "use_heuristic": False,
        "norm_obs": False,
        "constraints_type": 0,
        "chronics_split": cli.chronics_split,
        "eval_chronics_split": cli.chronics_split,
        "seed": cli.seed,
    })

    print(f"Baseline eval: env={cli.env_id} diff={cli.difficulty} action={cli.action_type} "
          f"split={cli.chronics_split} episodes={cli.episodes} workers={cli.workers}")
    for agent_name in cli.agents:
        eval_one_agent(agent_name, env_args, cli.episodes, cli.workers, cli.seed)


if __name__ == "__main__":
    main()
