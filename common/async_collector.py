"""Asynchronous (barrier-free) rollout collection for on-policy algorithms.

Why
---
`gym.vector.AsyncVectorEnv.step()` waits for *all* sub-environments before returning.  With the
RL2Grid heuristics (idle / recovery) one agent step can trigger anywhere from 0 to thousands of
internal grid2op steps, so every vector step waits for the slowest of `n_envs` workers and most
CPU time is spent idling.

How
---
Each sub-environment lives in its own worker process.  The collector keeps every worker busy:
as soon as *any* worker returns a transition, the policy is queried for the workers that are
ready (batched) and they are stepped again.  The only synchronisation point left is the end of a
rollout (once `n_transitions` have been dispatched we wait for the in-flight ones), which keeps
the data strictly on-policy for PPO.

Each worker therefore contributes a contiguous trajectory segment of *variable* length; the
caller computes GAE per segment (see `compute_segment_gae`).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from copy import deepcopy
from multiprocessing.connection import wait
from typing import Callable, Dict, List, Optional

import numpy as np
import torch as th


def _worker(remote, parent_remote, env_fn: Callable) -> None:
    parent_remote.close()
    env = None
    try:
        env = env_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                final_obs = None
                if terminated or truncated:  # autoreset, mirroring gymnasium 0.29 AsyncVectorEnv
                    final_obs = obs
                    obs, _ = env.reset()
                cost = float(info.get("cost", 0.0))  # CMDP wrappers only; taken before the autoreset
                raw_cost = float(info.get("raw_cost", cost))
                remote.send(("ok", (obs, float(reward), bool(terminated), bool(truncated), final_obs, cost, raw_cost)))
            elif cmd == "reset":
                obs, _ = env.reset(seed=data)
                remote.send(("ok", obs))
            elif cmd == "spaces":
                remote.send(("ok", (env.observation_space, env.action_space)))
            elif cmd == "get_attr":
                remote.send(("ok", getattr(env, data)))
            elif cmd == "set_attr":
                name, val = data
                setattr(env, name, val)
                remote.send(("ok", None))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:  # forward the traceback to the main process
        remote.send(("error", traceback.format_exc()))
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        remote.close()


class AsyncCollector:
    """Drop-in replacement for the parts of AsyncVectorEnv used by RL2Grid's PPO."""

    def __init__(self, env_fns: List[Callable]):
        # fork: env_fns are closures (not picklable) exactly as with gymnasium's AsyncVectorEnv
        ctx = mp.get_context("fork")
        self.num_envs = len(env_fns)
        self.remotes, self.processes = [], []
        for fn in env_fns:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, parent, fn), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(p)
        self.single_observation_space, self.single_action_space = self._call(0, "spaces")
        # Continuous actions must be clipped to the Box before stepping, otherwise grid2op flags
        # them as ambiguous and silently applies do-nothing (same fix as the synchronous path).
        space = self.single_action_space
        self._clip = (space.low, space.high) if hasattr(space, "low") and hasattr(space, "high") else None
        self.obs = None  # latest observation of every worker, shape (n_envs, obs_dim)
        self.n_transitions_total = 0

    # ------------------------------------------------------------------ helpers
    def _recv(self, i: int):
        status, payload = self.remotes[i].recv()
        if status == "error":
            raise RuntimeError(f"Async env worker {i} crashed:\n{payload}")
        return payload

    def _call(self, i: int, cmd: str, data=None):
        self.remotes[i].send((cmd, data))
        return self._recv(i)

    def reset(self, seed=None):
        for i, r in enumerate(self.remotes):
            r.send(("reset", None if seed is None else seed + i))
        # Cast to the declared dtype (float32) like AsyncVectorEnv's shared memory does; NormalizeObservation
        # emits float64, which the off-policy networks would reject. Later writes into self.obs keep this dtype.
        dtype = getattr(self.single_observation_space, "dtype", np.float32) or np.float32
        self.obs = np.stack([self._recv(i) for i in range(self.num_envs)]).astype(dtype)
        return self.obs, {}

    def _drain_pending(self) -> None:
        """Finish in-flight off-policy steps (kept for the next collect) so the pipes are idle."""
        for i in list(getattr(self, "_pending", {})):
            next_obs, reward, terminated, truncated, final_obs, _, _ = self._recv(i)
            obs, action = self._pending.pop(i)
            real_next = final_obs if final_obs is not None else next_obs
            self._carry.append((obs, real_next, action, reward, terminated))
            self.obs[i] = next_obs

    def get_attr(self, name: str) -> list:
        """Same contract as gymnasium's VectorEnv.get_attr (e.g. obs_rms for the evaluator)."""
        self._drain_pending()  # a pending step reply would otherwise be read as the attribute
        return [self._call(i, "get_attr", name) for i in range(self.num_envs)]

    def set_attr(self, name: str, values) -> None:
        """Mirror VectorEnv.set_attr (accepts a per-worker sequence or a single value to broadcast)."""
        self._drain_pending()
        if isinstance(values, (list, tuple)) and len(values) == self.num_envs:
            for i in range(self.num_envs):
                self._call(i, "set_attr", (name, deepcopy(values[i])))
        else:
            for i in range(self.num_envs):
                self._call(i, "set_attr", (name, deepcopy(values)))

    def close(self) -> None:
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    # ------------------------------------------------------------------ rollout
    @th.no_grad()
    def collect(self, agent, n_transitions: int, device: th.device, with_cost: bool = False) -> Dict[str, list]:
        """Collect exactly `n_transitions` on-policy transitions without per-step barriers.

        Returns per-worker segments (lists of tensors) plus the bootstrap value of each
        worker's last observation. `with_cost` (LagrPPO) also records the CMDP cost and the
        cost-critic values/bootstrap.
        """
        if self.obs is None:
            self.reset()
        n = self.num_envs
        seg = {k: [[] for _ in range(n)] for k in ("obs", "actions", "logprobs", "values", "rewards",
                                                    "terminations", "dones", "costs", "raw_costs", "cost_values")}
        ready = list(range(n))
        in_flight = set()
        dispatched = 0
        idle_waits = 0

        while dispatched < n_transitions or in_flight:
            # 1) dispatch every ready worker (batched policy query), never exceeding the budget
            budget = n_transitions - dispatched
            if ready and budget > 0:
                idx = ready[:budget]
                ready = ready[budget:]
                obs_t = th.as_tensor(self.obs[idx], dtype=th.float32, device=device)
                action, logprob, _ = agent.get_action(obs_t)
                value = agent.get_value(obs_t).flatten()
                cost_value = agent.get_cost_value(obs_t).flatten() if with_cost else None
                act_np = action.cpu().numpy()
                if self._clip is not None:  # stored action (for the log-prob) stays unclipped, as in sync PPO
                    act_np = np.clip(act_np, *self._clip)
                for k, i in enumerate(idx):
                    seg["obs"][i].append(obs_t[k])
                    seg["actions"][i].append(action[k])
                    seg["logprobs"][i].append(logprob[k])
                    seg["values"][i].append(value[k])
                    if with_cost: seg["cost_values"][i].append(cost_value[k])
                    self.remotes[i].send(("step", act_np[k]))
                    in_flight.add(i)
                dispatched += len(idx)
            # 2) block until at least one worker has a result, then drain all available results
            if not in_flight:
                continue
            conns = wait([self.remotes[i] for i in in_flight])
            idle_waits += 1
            for conn in conns:
                i = self.remotes.index(conn)
                next_obs, reward, terminated, truncated, final_obs, cost, raw_cost = self._recv(i)
                in_flight.discard(i)
                seg["rewards"][i].append(reward)
                seg["costs"][i].append(cost)
                seg["raw_costs"][i].append(raw_cost)
                seg["terminations"][i].append(int(terminated))
                seg["dones"][i].append(int(terminated or truncated))
                self.obs[i] = next_obs
                if dispatched < n_transitions:
                    ready.append(i)

        # bootstrap values for the last observation of every worker that collected data
        last_obs = th.as_tensor(self.obs, dtype=th.float32, device=device)
        boot = agent.get_value(last_obs).flatten()
        self.n_transitions_total += dispatched
        seg["bootstrap"] = boot
        if with_cost: seg["cost_bootstrap"] = agent.get_cost_value(last_obs).flatten()
        seg["n_waits"] = idle_waits
        return seg


    # ------------------------------------------------------------- off-policy
    def collect_transitions(self, act_fn: Callable, n: int) -> Dict[str, np.ndarray]:
        """Return exactly `n` transitions for a replay buffer, without any barrier.

        Off-policy algorithms (DQN / SAC / TD3) do not need the data to come from one policy, so
        workers are never drained: whatever is still in flight when `n` transitions are ready keeps
        simulating while the caller trains, and is returned by the next call. Results beyond `n`
        that arrive together are carried over, so no transition is ever dropped.

        `act_fn(obs_batch) -> actions` is called on the observations of the workers that are ready.
        """
        if self.obs is None:
            self.reset()
        if not hasattr(self, "_pending"):
            self._pending: Dict[int, tuple] = {}   # worker -> (obs, executed action)
            self._carry: List[tuple] = []          # finished transitions not yet returned
        out = self._carry
        self._carry = []
        while len(out) < n:
            ready = [i for i in range(self.num_envs) if i not in self._pending]
            if ready:
                obs_batch = self.obs[ready]
                actions = np.asarray(act_fn(obs_batch))
                if self._clip is not None:
                    actions = np.clip(actions, *self._clip)
                for k, i in enumerate(ready):
                    self._pending[i] = (obs_batch[k].copy(), actions[k])
                    self.remotes[i].send(("step", actions[k]))
            for conn in wait([self.remotes[i] for i in self._pending]):
                i = self.remotes.index(conn)
                next_obs, reward, terminated, truncated, final_obs, _, _ = self._recv(i)
                obs, action = self._pending.pop(i)
                real_next = final_obs if final_obs is not None else next_obs
                out.append((obs, real_next, action, reward, terminated))
                self.obs[i] = next_obs
        out, self._carry = out[:n], out[n:]
        self.n_transitions_total += n
        o, no, a, r, t = zip(*out)
        dt = self.obs.dtype
        return {"obs": np.stack(o).astype(dt), "next_obs": np.stack(no).astype(dt), "actions": np.stack(a),
                "rewards": np.asarray(r, dtype=np.float32), "terminations": np.asarray(t, dtype=np.float32)}


def _gae(values, rewards, terms, dones, bootstrap, gamma: float, gae_lambda: float):
    L = len(values)
    adv = th.zeros(L, device=values.device)
    lastgaelam = 0.0
    for t in reversed(range(L)):
        nextvalue = bootstrap if t == L - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalue * (1 - terms[t]) - values[t]
        adv[t] = lastgaelam = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
    return adv


def compute_segment_gae(seg: Dict[str, list], gamma: float, gae_lambda: float, device: th.device,
                        with_cost: bool = False, cost_gamma: Optional[float] = None):
    """Per-worker GAE on variable-length segments; returns flat batch tensors.

    Uses exactly the same recursion as RL2Grid's synchronous PPO:
        delta_t = r_t + gamma * V_{t+1} * (1 - term_t) - V_t
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    where V_{t+1} for the last step of a segment is the value of the worker's current observation.
    `with_cost` (LagrPPO) runs the same recursion on the costs with the cost critic (using `cost_gamma` if given).
    """
    cg = gamma if cost_gamma is None else cost_gamma
    keys = ["obs", "actions", "logprobs", "values", "advantages", "returns"]
    if with_cost: keys += ["cost_values", "cost_advantages", "cost_returns"]
    out = {k: [] for k in keys}
    f32 = dict(dtype=th.float32, device=device)
    for i in range(len(seg["obs"])):
        if len(seg["obs"][i]) == 0:
            continue
        terms = th.tensor(seg["terminations"][i], **f32)
        dones = th.tensor(seg["dones"][i], **f32)
        values = th.stack(seg["values"][i])
        adv = _gae(values, th.tensor(seg["rewards"][i], **f32), terms, dones, seg["bootstrap"][i], gamma, gae_lambda)
        out["obs"].append(th.stack(seg["obs"][i]))
        out["actions"].append(th.stack(seg["actions"][i]))
        out["logprobs"].append(th.stack(seg["logprobs"][i]))
        out["values"].append(values)
        out["advantages"].append(adv)
        out["returns"].append(adv + values)
        if with_cost:
            cvalues = th.stack(seg["cost_values"][i])
            cadv = _gae(cvalues, th.tensor(seg["costs"][i], **f32), terms, dones, seg["cost_bootstrap"][i],
                        cg, gae_lambda)
            out["cost_values"].append(cvalues)
            out["cost_advantages"].append(cadv)
            out["cost_returns"].append(cadv + cvalues)
    return {k: th.cat(v) for k, v in out.items()}
