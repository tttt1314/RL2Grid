from common.imports import *

class ConstrainedFailureGridOp(gym.Wrapper):
    def __init__(self, env, fail_absorbing_cost: float = 0.0, cost_gamma: float = 0.9):
        super().__init__(env)
        self.fail_absorbing_cost = float(fail_absorbing_cost)
        self.cost_gamma = float(cost_gamma)

    def step(self, gym_action: Union[int, List[float]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment using the given gym action, then return a cost function in the info.

        Args:
            gym_action: The action to take in the gym environment.

        Returns:
            A tuple containing the observation, reward, done flag, truncation flag, and additional info (with the cost).
        """
        next_obs, reward, terminated, truncated, info = super().step(gym_action)
        
        failed = self.init_env.nb_time_step < self.init_env.chronics_handler.max_episode_duration()
        raw_cost = 1.0 if (np.logical_or(terminated, truncated) and failed) else 0.0
        info['raw_cost'] = raw_cost
        info['cost'] = int(raw_cost)
        
        return (
            next_obs, 
            reward, 
            terminated, 
            truncated,  # Truncation is typically False in g2o envs
            info    
        )

class ConstrainedOverloadGridOp(gym.Wrapper):
    def __init__(self, env, fail_absorbing_cost: float = 0.0, cost_gamma: float = 0.9):
        super().__init__(env)
        self.fail_absorbing_cost = float(fail_absorbing_cost)
        self.cost_gamma = float(cost_gamma)

    def step(self, gym_action: Union[int, List[float]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment using the given gym action, then return a cost function in the info.

        Args:
            gym_action: The action to take in the gym environment.

        Returns:
            A tuple containing the observation, reward, done flag, truncation flag, and additional info (with the cost).
        """
        next_obs, reward, terminated, truncated, info = super().step(gym_action)

        obs = self.init_env.current_obs
        forced = obs.time_next_maintenance == 0     # lines currently under maintenance
        attacked = info.get('opponent_attack_line')
        if attacked is not None: forced = forced | attacked
        n_disconnections = int(np.sum(~obs.line_status & ~forced))
        n_overloads = 0     # If it's game over and the grid is disconnected (i..e, n_disconnections = n_lines), then we cannot get the thermal limit and we don't have to compute n_overloads

        done = bool(np.logical_or(terminated, truncated))
        if not done:
            ampere_flows = np.abs(self.init_env.backend.get_line_flow())
            thermal_limits = np.abs(self.init_env.get_thermal_limit())        
            margin = thermal_limits - ampere_flows
            n_overloads = len(margin[margin < 0])
        
        raw_cost = float(n_disconnections + n_overloads)
        info['raw_cost'] = raw_cost
        max_steps = self.init_env.chronics_handler.max_episode_duration()
        if self.fail_absorbing_cost > 0.0 and done and self.init_env.nb_time_step < max_steps:
            remaining = max(0, int(max_steps) - int(self.init_env.nb_time_step))
            if self.cost_gamma < 1.0:
                c_term = self.fail_absorbing_cost * (1.0 - self.cost_gamma ** remaining) / (1.0 - self.cost_gamma)
            else:
                c_term = self.fail_absorbing_cost * float(remaining)
            info['cost'] = raw_cost + c_term
        else:
            info['cost'] = raw_cost

        return (
            next_obs, 
            reward, 
            terminated, 
            truncated,  # Truncation is typically False in g2o envs
            info    
        )
