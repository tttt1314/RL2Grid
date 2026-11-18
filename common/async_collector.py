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
from multiprocessing.connection import wait
from typing import Callable, Dict, List

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
                obs, reward, terminated, truncated, _ = env.step(data)
                final_obs = None
                if terminated or truncated:  # autoreset, mirroring gymnasium 0.29 AsyncVectorEnv
                    final_obs = obs
                    obs, _ = env.reset()
                remote.send(("ok", (obs, float(reward), bool(terminated), bool(truncated), final_obs)))
            elif cmd == "reset":
                obs, _ = env.reset(seed=data)
                remote.send(("ok", obs))
            elif cmd == "spaces":
                remote.send(("ok", (env.observation_space, env.action_space)))
            elif cmd == "get_attr":
                remote.send(("ok", getattr(env, data)))
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
        self.obs = np.stack([self._recv(i) for i in range(self.num_envs)])
        return self.obs, {}

    def _drain_pending(self) -> None:
        """Finish in-flight off-policy steps (kept for the next collect) so the pipes are idle."""
        for i in list(getattr(self, "_pending", {})):
            next_obs, reward, terminated, truncated, final_obs = self._recv(i)
            obs, action = self._pending.pop(i)
            real_next = final_obs if final_obs is not None else next_obs
            self._carry.append((obs, real_next, action, reward, terminated))
            self.obs[i] = next_obs

    def get_attr(self, name: str) -> list:
        """Same contract as gymnasium's VectorEnv.get_attr (e.g. obs_rms for the evaluator)."""
        self._drain_pending()  # a pending step reply would otherwise be read as the attribute
        return [self._call(i, "get_attr", name) for i in range(self.num_envs)]

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
    def collect(self, agent, n_transitions: int, device: th.device) -> Dict[str, list]:
        """Collect exactly `n_transitions` on-policy transitions without per-step barriers.

        Returns per-worker segments (lists of tensors) plus the bootstrap value of each
        worker's last observation.
        """
        if self.obs is None:
            self.reset()
        n = self.num_envs
        seg = {k: [[] for _ in range(n)] for k in ("obs", "actions", "logprobs", "values", "rewards",
                                                    "terminations", "dones")}
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
                act_np = action.cpu().numpy()
                if self._clip is not None:  # stored action (for the log-prob) stays unclipped, as in sync PPO
                    act_np = np.clip(act_np, *self._clip)
                for k, i in enumerate(idx):
                    seg["obs"][i].append(obs_t[k])
                    seg["actions"][i].append(action[k])
                    seg["logprobs"][i].append(logprob[k])
                    seg["values"][i].append(value[k])
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
                next_obs, reward, terminated, truncated, final_obs = self._recv(i)
                in_flight.discard(i)
                seg["rewards"][i].append(reward)
                seg["terminations"][i].append(int(terminated))
                seg["dones"][i].append(int(terminated or truncated))
                self.obs[i] = next_obs
                if dispatched < n_transitions:
                    ready.append(i)

        # bootstrap values for the last observation of every worker that collected data
        boot = agent.get_value(th.as_tensor(self.obs, dtype=th.float32, device=device)).flatten()
        self.n_transitions_total += dispatched
        seg["bootstrap"] = boot
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
                next_obs, reward, terminated, truncated, final_obs = self._recv(i)
                obs, action = self._pending.pop(i)
                real_next = final_obs if final_obs is not None else next_obs
                out.append((obs, real_next, action, reward, terminated))
                self.obs[i] = next_obs
        out, self._carry = out[:n], out[n:]
        self.n_transitions_total += n
        o, no, a, r, t = zip(*out)
        return {"obs": np.stack(o), "next_obs": np.stack(no), "actions": np.stack(a),
                "rewards": np.asarray(r, dtype=np.float32), "terminations": np.asarray(t, dtype=np.float32)}


def compute_segment_gae(seg: Dict[str, list], gamma: float, gae_lambda: float, device: th.device):
    """Per-worker GAE on variable-length segments; returns flat batch tensors.

    Uses exactly the same recursion as RL2Grid's synchronous PPO:
        delta_t = r_t + gamma * V_{t+1} * (1 - term_t) - V_t
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    where V_{t+1} for the last step of a segment is the value of the worker's current observation.
    """
    out = {k: [] for k in ("obs", "actions", "logprobs", "values", "advantages", "returns")}
    for i in range(len(seg["obs"])):
        L = len(seg["obs"][i])
        if L == 0:
            continue
        values = th.stack(seg["values"][i])
        rewards = th.tensor(seg["rewards"][i], dtype=th.float32, device=device)
        terms = th.tensor(seg["terminations"][i], dtype=th.float32, device=device)
        dones = th.tensor(seg["dones"][i], dtype=th.float32, device=device)
        adv = th.zeros(L, device=device)
        lastgaelam = 0.0
        for t in reversed(range(L)):
            nextvalue = seg["bootstrap"][i] if t == L - 1 else values[t + 1]
            delta = rewards[t] + gamma * nextvalue * (1 - terms[t]) - values[t]
            adv[t] = lastgaelam = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
        out["obs"].append(th.stack(seg["obs"][i]))
        out["actions"].append(th.stack(seg["actions"][i]))
        out["logprobs"].append(th.stack(seg["logprobs"][i]))
        out["values"].append(values)
        out["advantages"].append(adv)
        out["returns"].append(adv + values)
    return {k: th.cat(v) for k, v in out.items()}
