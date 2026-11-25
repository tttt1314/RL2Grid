from common.imports import *
from common.utils import str2bool

def get_env_args() -> Namespace:
    """
    Parse and return the command-line arguments for configuring the environment.

    Returns:
        Namespace: A namespace containing the parsed arguments.
    """
    parser = ap.ArgumentParser()

    # Settings
    parser.add_argument("--env-id", type=str, default="bus14", help="ID of the grid2op environment")
    parser.add_argument("--n-envs", type=int, default=50, help="Number of parallel envs to run")
    parser.add_argument("--action-type", type=str, default="topology", choices=["topology", "redispatch"], help="Type of environment: topology (discrete) or redispatch (continuous)")
    parser.add_argument("--difficulty", type=int, default=0, help="Higher difficulty means bigger action spaces")
    parser.add_argument("--n1-reward", type=str2bool, default=False, help="Toggles N1 contintency analysis as an additional reward")

    # Scenarios
    parser.add_argument("--env-config-path", type=str, default="scenario.json", help="Path to environment configuration file")
    
    parser.add_argument("--gamma", type=float, default=.9, help="Discount factor")

    # Normalization
    parser.add_argument("--norm-obs", type=str2bool, default=True, help="Toggle normalize observations")
    parser.add_argument("--use-heuristic", type=str2bool, default=False, help="Toggles heuristics for base operations")
    parser.add_argument("--heuristic-type", type=str, default="idle", choices=["idle", "reconnect"], help="Select the type of heuristic to use: idle or reconnect")

    parser.add_argument("--optimize-mem", type=str2bool, default=True, help="Whether to load data chunks upon resets (True), or the whole dataset once (False)")
    parser.add_argument("--constraints-type", type=int, default=0, choices=[0, 1, 2], help="Select the type of constraints to use: no constraints (0), failure constraints (1), overloads constraints (2)")
    parser.add_argument("--chronics-split", type=str, default="all", choices=["all", "train", "val", "test"],
                        help="Deterministic chronics split for training/eval (all=original, train=70%, val=15%, test=15%)")
    parser.add_argument("--eval-chronics-split", type=str, default="", choices=["", "all", "train", "val", "test"],
                        help="Optional chronics split override for evaluation environments (empty=same as --chronics-split)")
    parser.add_argument("--keep-disc-penalty", type=str2bool, default=False,
                        help="Keep OverloadReward disconnection penalty (-1.0) even when constraints-type != 0")
    parser.add_argument("--fail-absorbing-cost", type=float, default=0.0,
                        help="Per-step absorbing cost c_max charged for remaining horizon on premature failure (TLO/RD1)")
    parser.add_argument("--cost-gamma", type=float, default=-1.0,
                        help="Independent discount factor for CMDP cost critic and absorbing terminal cost (<0 uses --gamma)")

    # Parse the arguments
    params, _ = parser.parse_known_args()

    return params
