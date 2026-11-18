from collections import deque
from copy import deepcopy

from common.imports import *
from common.logger import Logger
from .utils import auxiliary_make_env


import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait


def _eval_episode(env, model, obs_rms, use_heuristic: bool, n_tags: int, max_steps: int, cmdp: bool):
    """One deterministic evaluation episode; same bookkeeping as Evaluator.evaluate."""
    if obs_rms is not None: env.obs_rms = deepcopy(obs_rms)
    obs, info = env.reset()
    ep_rewards, ep_cost = np.zeros(n_tags), 0.
    if use_heuristic: ep_rewards += list(info['rewards'].values())
    while True:
        with th.no_grad():
            action = model.get_eval_action(th.tensor(obs, dtype=th.float)).detach().cpu().numpy()
        obs, _, _, _, info = env.step(action)
        if use_heuristic: ep_rewards = np.array(list(info['rewards'].values()), dtype=float)  # already cumulative
        else: ep_rewards += list(info['rewards'].values())
        if cmdp: ep_cost += info['cost']
        if "episode" in info:
            return env.init_env.nb_time_step / max_steps, ep_rewards, ep_cost


def _eval_worker(remote, parent_remote, args, idx: int, n_tags: int) -> None:
    parent_remote.close()
    th.set_num_threads(1)
    try:
        env = auxiliary_make_env(args, eval_env=True, idx=idx)[0]   # idx -> own seed -> own chronic order
        if args.use_heuristic: env.set_n_rewards(n_tags)
        max_steps = env.init_env.chronics_handler.max_episode_duration()
        remote.send(("ok", None))
        while True:
            cmd, data = remote.recv()
            if cmd == "close": break
            model, obs_rms, n_ep, cmdp = data
            remote.send(("ok", [_eval_episode(env, model, obs_rms, args.use_heuristic, n_tags, max_steps, cmdp)
                                for _ in range(n_ep)]))
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:
        import traceback
        remote.send(("error", traceback.format_exc()))


class EvalPool:
    """Persistent eval-env workers; each runs whole episodes with a copy of the policy (no per-step sync)."""

    def __init__(self, args, n_workers: int, n_tags: int, base_idx: int = 1000):
        ctx = mp.get_context("fork")
        self.remotes, self.procs = [], []
        for i in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_eval_worker, args=(child, parent, args, base_idx + i, n_tags), daemon=True)
            p.start(); child.close()
            self.remotes.append(parent); self.procs.append(p)
        for r in self.remotes: self._check(r.recv())

    @staticmethod
    def _check(msg):
        status, payload = msg
        if status == "error": raise RuntimeError(f"eval worker crashed:\n{payload}")
        return payload

    def run(self, model, obs_rms, n_episodes: int, cmdp: bool) -> list:
        n = len(self.remotes)
        split = [n_episodes // n + (1 if i < n_episodes % n else 0) for i in range(n)]
        model_cpu = deepcopy(model).cpu()
        busy = []
        for r, k in zip(self.remotes, split):
            if k: r.send(("run", (model_cpu, obs_rms, k, cmdp))); busy.append(r)
        out = []
        for r in busy: out += self._check(r.recv())
        return out

    def close(self) -> None:
        for r in self.remotes:
            try: r.send(("close", None))
            except (BrokenPipeError, OSError): pass
        for p in self.procs: p.join(timeout=5)


class Evaluator:
    """Evaluator class for evaluating a reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, args: Dict[str, Any], logger: Logger, device: th.device, envs: gym.Env = None) -> None:
        """Initialize the Evaluator with the given arguments, logger, and device.

        Args:
            args: Arguments containing environment configuration.
            logger: Logger for storing evaluation metrics.
            device: Device to run the model on.
        """
 
        self.env = auxiliary_make_env(args, eval_env=True)[0]  # Initialize synchronized vector environment
        self.max_steps = self.env.init_env.chronics_handler.max_episode_duration()  # Get max episode duration

        self.logger = logger  # Logger for evaluation metrics
        self.device = device  # Device for model inference

        # Fix n° rewards based on the env specs (for simplifying logging ops)
        # The reward returned by the env is an increasing survival reward
        self.reward_tags = ['Redispatch Reward', 'Line Margin Reward', 'Overload Reward']   
        if args.action_type == 'topology': self.reward_tags += ['Topology Reward']      
        if args.n1_reward: self.reward_tags += ['N1 Reward']
        self.use_heuristic = args.use_heuristic
        # --eval-episodes (default 3 = original): more episodes -> less noisy survival estimates
        self.eval_episodes = getattr(args, 'eval_episodes', 3)
        self.train_envs = envs if args.norm_obs else None   # source of the observation statistics
        if self.use_heuristic: self.env.set_n_rewards(len(self.reward_tags))
        # --eval-workers (default 1 = original sequential loop): run the evaluation episodes in parallel
        n_workers = min(getattr(args, 'eval_workers', 1), self.eval_episodes)
        self.pool = EvalPool(args, n_workers, len(self.reward_tags)) if n_workers > 1 else None
            
    def _reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the eval env with the training observation statistics."""
        if self.train_envs is not None: self.env.obs_rms = deepcopy(self.train_envs.get_attr('obs_rms')[0])
        return self.env.reset()

    def evaluate(self, glob_step: int, model: object, eval_ep: int = None) -> None:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """
        
        eval_ep = eval_ep or self.eval_episodes
        if self.pool is not None:
            obs_rms = deepcopy(self.train_envs.get_attr('obs_rms')[0]) if self.train_envs is not None else None
            results = self.pool.run(model, obs_rms, eval_ep, cmdp=False)
            avg_survival = sum(r[0] for r in results) / eval_ep
            avg_return = [sum(col) / eval_ep for col in zip(*[r[1] for r in results])]
            if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, self.reward_tags)
            print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}")
            return

        ep_survivals: Deque[float] = deque(maxlen=eval_ep)  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))

        obs, info = self._reset()

        if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        while len(ep_survivals) < eval_ep:
            action = model.get_eval_action(th.tensor(obs, dtype=th.float).to(self.device)).detach().cpu().numpy()
            next_obs, _, _, _, info = self.env.step(action)

            obs = next_obs

            if self.use_heuristic: ep_rewards = np.array(list(info['rewards'].values()), dtype=float)  # already cumulative
            else: ep_rewards += list(info['rewards'].values())
            # Record rewards for plotting purposes
            if "episode" in info:   # Denote end of an episode
                ep_survivals.append(self.env.init_env.nb_time_step/self.max_steps)
                ep_returns.append(ep_rewards)
                obs, _ = self._reset()
                ep_rewards = np.zeros(len(self.reward_tags))

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals)/eval_ep
        avg_return = [sum(r)/eval_ep for r in zip(*ep_returns)]

        # Log the metrics if logger is available
        if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, self.reward_tags)

        print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}")

class CMDPEvaluator(Evaluator):
    """Evaluator class for evaluating a constrained reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, args: Dict[str, Any], logger: Logger, device: th.device, envs: gym.Env = None) -> None:
        super().__init__(args, logger, device, envs)

    def evaluate(self, glob_step: int, model: object, eval_ep: int = None) -> None:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """
        
        eval_ep = eval_ep or self.eval_episodes
        if self.pool is not None:
            obs_rms = deepcopy(self.train_envs.get_attr('obs_rms')[0]) if self.train_envs is not None else None
            results = self.pool.run(model, obs_rms, eval_ep, cmdp=True)
            avg_survival = sum(r[0] for r in results) / eval_ep
            avg_return = [sum(col) / eval_ep for col in zip(*[r[1] for r in results])]
            avg_cost_return = [sum(r[2] for r in results) / eval_ep]
            if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, avg_cost_return, self.reward_tags)
            print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}, cost_return={avg_cost_return}")
            return

        ep_survivals: Deque[float] = deque(maxlen=eval_ep)  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store returns of episodes
        ep_cost_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store cost returns of episodes
        ep_rewards = np.zeros(len(self.reward_tags))
        ep_costs = 0

        obs, info = self._reset()

        if self.use_heuristic: ep_rewards += list(info['rewards'].values())

        while len(ep_survivals) < eval_ep:
            action = model.get_eval_action(th.tensor(obs, dtype=th.float).to(self.device)).detach().cpu().numpy()
            next_obs, _, _, _, info = self.env.step(action)

            obs = next_obs
            if self.use_heuristic: ep_rewards = np.array(list(info['rewards'].values()), dtype=float)  # already cumulative
            else: ep_rewards += list(info['rewards'].values())
            ep_costs += info['cost']

            # Record rewards and cost for plotting purposes
            if "episode" in info:   # Denote end of an episode
                ep_survivals.append(self.env.init_env.nb_time_step/self.max_steps)
                ep_returns.append(ep_rewards)
                ep_cost_returns.append(ep_costs)

                obs, _ = self._reset()
                ep_rewards = np.zeros(len(self.reward_tags))
                ep_costs = 0

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals)/eval_ep
        avg_return = [sum(r)/eval_ep for r in zip(*ep_returns)]
        avg_cost_return = [sum(ep_cost_returns)/eval_ep]

        # Log the metrics if logger is available
        if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return, avg_cost_return, self.reward_tags)

        print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}, cost_return={avg_cost_return}")
