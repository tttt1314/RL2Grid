import os
# Must be set before numpy/torch load OpenBLAS. With the default (one thread per core) the idle OpenBLAS
# threads of the main process busy-wait after every policy forward and starve the env workers.
# bus14 PPO, 32 vCPU: 32 threads ~330 SPS, 1: 352, 4: 404, 8: 412 (best trade-off with the PPO update).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

from multiprocessing import Manager
from time import time

from grid2op.gym_compat import BoxGymActSpace, DiscreteActSpace # if we import gymnasium, GymEnv will convert to Gymnasium!   

from alg.dqn.core import DQN
from alg.lagr_ppo.core import LagrPPO
from alg.ppo.core import PPO
from alg.sac.core import SAC
from alg.td3.core import TD3
from common.async_collector import AsyncCollector
from common.checkpoint import DQNCheckpoint, LagrPPOCheckpoint, PPOCheckpoint, SACCheckpoint, TD3Checkpoint
from common.imports import *
from common.utils import set_random_seed, set_torch, str2bool
from env.config import get_env_args
from env.utils import auxiliary_make_env


# Dictionary mapping algorithm names to their corresponding classes
ALGORITHMS: Dict[str, Type[Any]] = {'DQN': DQN, 'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'LAGRPPO': LagrPPO}

def main(args: Namespace) -> None:
    """
    Main function to run the RL algorithms based on the provided arguments.

    Args:
        args (Namespace): Command line arguments parsed by argparse.

    Raises:
        AssertionError: If time limit exceeds 2800 minutes or if number of environments is less than 1.
        AssertionError: If the specified algorithm is not supported.
    """
    assert args.time_limit <= 2800, f"Invalid time limit: {args.time_limit}. Timeout limit is : 2800"
    start_time = time()
    
    # Update args with environment arguments
    args = ap.Namespace(**vars(args), **vars(get_env_args()))
    assert args.n_envs >= 1, f"Invalid n° of environments: {args.n_envs}. Must be >= 1"
    
    alg = args.alg.upper()
    assert alg in ALGORITHMS.keys(), f"Unsupported algorithm: {alg}. Supported algorithms are: {ALGORITHMS}"
    if (alg == "LAGRPPO" and args.constraints_type == 0) or (alg != "LAGRPPO" and args.constraints_type in [1, 2]):
        raise ValueError("Check the constrained version of the alg/env!")

    run_name = args.resume_run_name if args.resume_run_name \
        else f"{args.alg}_{args.env_id}_{"T" if args.action_type == "topology" else "R"}_{args.seed}_{args.difficulty}_{"H" if args.use_heuristic else ""}_{"I" if args.heuristic_type == "idle" else ""}_{"C1" if args.constraints_type == 1 else "C2" if args.constraints_type == 2 else ""}_{int(time())}_{np.random.randint(0, 50000)}"

    # Initialize the appropriate checkpoint based on the algorithm
    if alg == 'LAGRPPO': checkpoint = LagrPPOCheckpoint(run_name, args)
    elif alg == 'DQN': checkpoint = DQNCheckpoint(run_name, args)
    elif alg == 'PPO' : checkpoint = PPOCheckpoint(run_name, args)
    elif alg == 'SAC': checkpoint = SACCheckpoint(run_name, args)
    elif alg == 'TD3': checkpoint = TD3Checkpoint(run_name, args)
    else:
        pass  # This case should not occur due to earlier assertion

    # Set random seed and Torch configuration
    set_random_seed(args.seed)
    set_torch(args.n_threads, args.th_deterministic, args.cuda)
    
    # Resume run if checkpoint was resumed
    if checkpoint.resumed: args = checkpoint.loaded_run['args']

    # Create multiple async environments for parallel processing
    main_gym_env, main_g2o_env = auxiliary_make_env(args, checkpoint.resumed)

    with Manager() as manager:
        if args.action_type == "topology":
            if getattr(args, "shared_action_space", False):
                # Original behaviour: one Manager process serves every worker's action lookup over IPC,
                # which serialises all n_envs workers on a single process (main bottleneck on bus14).
                print("Initializing the distributed action 'mapper'... (takes a while with big action spaces)")
                shared_action_space = manager.list(main_gym_env.action_space.converter.all_actions)
            else:
                # Workers are forked, so they inherit this list directly: same actions, no per-step IPC.
                shared_action_space = list(main_gym_env.action_space.converter.all_actions)

        def make_vec_subprocess(idx):
            if args.action_type == "topology":
                action_space = DiscreteActSpace(main_g2o_env.action_space,
                                                    action_list=shared_action_space)
                return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx, action_space=action_space)[0]
            
            return auxiliary_make_env(args, resume_run=checkpoint.resumed, idx=idx)[0]
            
        env_fns = [lambda i=i: make_vec_subprocess(i) for i in range(args.n_envs)]
        if getattr(args, "async_collect", False):
            if alg not in ("PPO", "DQN", "SAC", "TD3"):
                raise NotImplementedError("--async-collect is implemented for PPO, DQN, SAC and TD3 (not LagrPPO)")
            envs = AsyncCollector(env_fns)
        else:
            envs = gym.vector.AsyncVectorEnv(env_fns)

        # Run the specified algorithm
        ALGORITHMS[alg](envs, run_name, start_time, args, checkpoint)
        
if __name__ == "__main__":
    parser = ap.ArgumentParser()

    # Cluster
    parser.add_argument("--time-limit", type=float, default=2790, help="Time limit")
    parser.add_argument("--checkpoint", type=str2bool, default=False, help="Toggles checkpoint.")
    parser.add_argument("--resume-run-name", type=str, default='', help="Run name to resume")
    parser.add_argument("--async-collect", type=str2bool, default=False,
                        help="Barrier-free rollout collection (PPO only); off = original AsyncVectorEnv")
    parser.add_argument("--eval-workers", type=int, default=1,
                        help="Processes for evaluation episodes (1 = original sequential evaluation in the main process)")
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="Episodes per evaluation (original: 3). Each full episode is up to 8064 steps.")
    parser.add_argument("--shared-action-space", type=str2bool, default=False,
                        help="Serve topology actions from a multiprocessing.Manager (original, slow); "
                             "default False gives each forked worker a local copy of the same action list")

    # Reproducibility
    parser.add_argument("--alg", type=str, default='PPO', help="Algorithm to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Logger
    parser.add_argument("--verbose", type=str2bool, default=True, help="Toggles prints")
    parser.add_argument("--exp-tag", type=str, default='', help="Tag for logging the experiment")
    parser.add_argument("--track", type=str2bool, default=False, help="Tag for logging the experiment")
    parser.add_argument("--wandb-project", type=str, default="", help="Wandb's project name.")
    parser.add_argument("--wandb-entity", type=str, default="", help="Entity (team) of wandb's project.")
    parser.add_argument("--wandb-mode", type=str, default="offline", help="Online or offline wandb mode.")

    # Torch
    parser.add_argument("--th-deterministic", type=str2bool, default=True, help="Enable deterministic in Torch.")
    parser.add_argument("--cuda", type=str2bool, default=False, help="Enable CUDA by default.")
    # 1 thread: the conda torch build uses OpenBLAS, and torch intra-op threads x OpenBLAS threads oversubscribe
    # the CPU (PPO update on bus14: 1 thread 11 s, 12 threads 69 s). Matmuls are still parallelised by OpenBLAS.
    parser.add_argument("--n-threads", type=int, default=1, help="Max number of torch threads.")

    main(parser.parse_known_args()[0])
