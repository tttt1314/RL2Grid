from time import time

from torch.nn.functional import softplus

from .agent import Agent
from .config import get_alg_args
from common.async_collector import compute_segment_gae
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import ConstrainedLogger
from env.eval import CMDPEvaluator

class LagrPPO:
    """Lagrangian implementation of the Lagrangian Proximal Policy Optimization (LagrPPO) implementation for training a constrained agent in a given environment: https://cdn.openai.com/safexp-short.pdf.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        """Init method for Lagrangian PPO

        Args:
            envs (gym.Env): The environments used for training.
            run_name (str): The name of the current training run.
            start_time (float): The time when training started.
            args (Dict[str, Any]): The command line arguments for configuration.
            ckpt (CheckpointSaver): The checkpoint handler for saving and loading training state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.n_steps % args.n_envs == 0, \
            f"Invalid train frequency (n_steps): {args.n_steps}. Must be multiple of n_envs {args.n_envs}"

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # Initialize the rollout, actor, critic, optimizer, and buffer
        batch_size = int(args.n_envs * args.n_steps)
        minibatch_size = int(batch_size // args.n_minibatches)
        n_rollouts = args.total_timesteps // batch_size
        init_rollout = 1 if not ckpt.resumed else ckpt.loaded_run['last_rollout']

        # Determine action space type
        continuous_actions = True if args.action_type == "redispatch" else False
        agent = Agent(envs, args, continuous_actions).to(device)

        if ckpt.resumed:
            agent.actor.load_state_dict(ckpt.loaded_run['actor'])
            agent.critic.load_state_dict(ckpt.loaded_run['critic'])
            agent.cost_critic.load_state_dict(ckpt.loaded_run['cost_critic'])

        actor_params = list(agent.actor.parameters())
        actor_optim = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(agent.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        cost_critic_optim = optim.Adam(agent.cost_critic.parameters(), lr=args.critic_lr, eps=1e-5)

        # Lagrangian setup
        lag_mul = th.tensor(args.lag_mul, requires_grad=True, device=device)
        lag_optim = optim.Adam([lag_mul], lr=args.lag_lr)
        cost_threshold = args.cost_threshold

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])
            cost_critic_optim.load_state_dict(ckpt.loaded_run['cost_critic_optim'])
            lag_mul.data = th.tensor(ckpt.loaded_run['lag_mul'], device=device)
            lag_optim.load_state_dict(ckpt.loaded_run['lag_optim'])
            cost_threshold = ckpt.loaded_run['cost_threshold']

        observations = th.zeros((args.n_steps, args.n_envs) + envs.single_observation_space.shape).to(device)
        actions = th.zeros((args.n_steps, args.n_envs) + 
                           envs.single_action_space.shape ).to(device)
        logprobs = th.zeros((args.n_steps, args.n_envs)).to(device)
        rewards = th.zeros((args.n_steps, args.n_envs)).to(device)
        costs = th.zeros((args.n_steps, args.n_envs)).to(device)
        dones = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        terminations = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        values = th.zeros((args.n_steps, args.n_envs)).to(device)
        cost_values = th.zeros((args.n_steps, args.n_envs)).to(device)

        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"
        logger = ConstrainedLogger(run_name, args) if args.track else None
        evaluator = CMDPEvaluator(args, logger, device, envs)
        max_steps = float(evaluator.max_steps)

        cost_gamma = getattr(args, "cost_gamma", -1.0)
        cost_gamma = args.gamma if cost_gamma < 0 else cost_gamma
        fail_abs = getattr(args, "fail_absorbing_cost", 0.0)
        dual_stat = getattr(args, "dual_stat", "window8k")
        lag_mode = getattr(args, "lag_update_mode", "adam")

        pid_i = 0.0
        pid_prev_jc = None
        pid_nu = max(0.0, float(args.lag_mul))
        if ckpt.resumed and "pid_state" in ckpt.loaded_run:
            ps = ckpt.loaded_run["pid_state"]
            pid_i = float(ps.get("pid_i", 0.0))
            pid_prev_jc = ps.get("pid_prev_jc", None)
            pid_nu = float(ps.get("pid_nu", pid_nu))

        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        ckpt.restore_obs_rms(envs, args)
        next_obs, _ = envs.reset()
        next_obs = th.tensor(next_obs).to(device)
        ep_cost_acc = np.zeros(args.n_envs, dtype=np.float64)
        ep_raw_cost_acc = np.zeros(args.n_envs, dtype=np.float64)
        ep_len_acc = np.zeros(args.n_envs, dtype=np.int64)
        raw_costs = np.zeros((args.n_steps, args.n_envs), dtype=np.float64)

        try:
            for iteration in range(init_rollout, n_rollouts + 1):
                finished_ep_costs, finished_ep_raw_costs, finished_ep_rates, finished_ep_lens = [], [], [], []
                # Annealing the rate if instructed to do so
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / n_rollouts
                    actor_optim.param_groups[0]['lr'] = frac * args.actor_lr
                    critic_optim.param_groups[0]['lr'] = frac * args.critic_lr

                if getattr(args, "async_collect", False):
                    # Barrier-free rollout (see common/async_collector.py), with the cost critic
                    t_roll = time()
                    seg = envs.collect(agent, batch_size, device, with_cost=True)
                    prev_step, global_step = global_step, global_step + batch_size
                    raw_seg = seg.get("raw_costs", seg["costs"])
                    for i in range(args.n_envs):
                        for c_t, rc_t, d_t in zip(seg["costs"][i], raw_seg[i], seg["dones"][i]):
                            ep_cost_acc[i] += c_t
                            ep_raw_cost_acc[i] += rc_t
                            ep_len_acc[i] += 1
                            if d_t:
                                ep_len = int(ep_len_acc[i])
                                raw_c = float(ep_raw_cost_acc[i])
                                finished_ep_costs.append(ep_cost_acc[i])
                                finished_ep_raw_costs.append(raw_c)
                                finished_ep_rates.append((raw_c + max(0.0, max_steps - ep_len) * fail_abs) / max_steps)
                                finished_ep_lens.append(ep_len)
                                ep_cost_acc[i] = 0.0
                                ep_raw_cost_acc[i] = 0.0
                                ep_len_acc[i] = 0
                    # The policy is fixed within a rollout: one eval per crossed eval_freq boundary
                    if global_step // args.eval_freq > prev_step // args.eval_freq:
                        evaluator.evaluate(global_step, agent)
                        if args.verbose:
                            print(f"SPS={int(global_step / (time() - start_time))} "
                                  f"rollout_s={time() - t_roll:.1f} waits={seg['n_waits']}")

                    k = int(8000 / args.n_envs)
                    costs_q_sum = float(sum(sum(c[-k:]) for c in seg["costs"]))
                    with th.no_grad():
                        batch = compute_segment_gae(seg, args.gamma, args.gae_lambda, device,
                                                    with_cost=True, cost_gamma=cost_gamma)
                    b_obs, b_logprobs, b_actions = batch["obs"], batch["logprobs"], batch["actions"]
                    b_advantages, b_returns, b_values = batch["advantages"], batch["returns"], batch["values"]
                    b_cost_advantages, b_cost_returns = batch["cost_advantages"], batch["cost_returns"]
                    b_cost_values = batch["cost_values"]
                else:
                    for step in range(0, args.n_steps):
                        global_step += args.n_envs
                        observations[step] = next_obs
                    
                        with th.no_grad():
                            action, logprob, _ = agent.get_action(next_obs)
                            value = agent.get_value(next_obs)
                            values[step] = value.flatten()
                            cost_value = agent.get_cost_value(next_obs)
                            cost_values[step] = cost_value.flatten()
                        actions[step] = action
                        logprobs[step] = logprob

                        next_obs, reward, next_terminations, next_truncations, infos = envs.step(np.clip(action.cpu().numpy(), envs.single_action_space.low, envs.single_action_space.high) if continuous_actions else action.cpu().numpy())

                        rewards[step] = th.tensor(reward).to(device).view(-1)
                        dones[step] = th.tensor(np.logical_or(next_terminations, next_truncations)).to(device)
                        terminations[step] = th.tensor(next_terminations).to(device)

                        real_next_obs = next_obs.copy()
                        cost = np.array(infos.get('cost', np.zeros(args.n_envs)), dtype=np.float64, copy=True)
                        raw_cost = np.array(infos.get('raw_cost', cost), dtype=np.float64, copy=True)
                        for idx, done in enumerate(dones[step]):
                            if done: 
                                real_next_obs[idx] = infos["final_observation"][idx]
                                fi = infos["final_info"][idx]
                                cost[idx] = fi['cost']   # When done, 'cost' is also in 'final_info'
                                raw_cost[idx] = fi.get('raw_cost', fi['cost'])
    
                        costs[step] = th.tensor(cost).to(device).view(-1)
                        raw_costs[step] = raw_cost
                        next_obs = th.tensor(next_obs).to(device)
                        real_next_obs = th.tensor(real_next_obs).to(device)

                        if global_step % args.eval_freq == 0:
                            evaluator.evaluate(global_step, agent)
                            if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")

                    costs_np = costs.detach().cpu().numpy()
                    dones_np = dones.detach().cpu().numpy()
                    for step_idx in range(args.n_steps):
                        ep_cost_acc += costs_np[step_idx]
                        ep_raw_cost_acc += raw_costs[step_idx]
                        ep_len_acc += 1
                        done_mask = dones_np[step_idx].astype(bool)
                        if done_mask.any():
                            for idx in np.where(done_mask)[0]:
                                ep_len = int(ep_len_acc[idx])
                                raw_c = float(ep_raw_cost_acc[idx])
                                finished_ep_costs.append(float(ep_cost_acc[idx]))
                                finished_ep_raw_costs.append(raw_c)
                                finished_ep_rates.append((raw_c + max(0.0, max_steps - ep_len) * fail_abs) / max_steps)
                                finished_ep_lens.append(ep_len)
                            ep_cost_acc[done_mask] = 0.0
                            ep_raw_cost_acc[done_mask] = 0.0
                            ep_len_acc[done_mask] = 0

                    costs_q = costs[-int(8000/args.n_envs):, :].flatten()
                    costs_q_sum = float(costs_q.sum().item())

                    # Bootstrap value if not done
                    with th.no_grad():
                        advantages = th.zeros_like(rewards).to(device)
                        cost_advantages = th.zeros_like(rewards).to(device)
                        lastgaelam, lastcostgaelam = 0, 0
                        for t in reversed(range(args.n_steps)):
                            if t == args.n_steps - 1:
                                nextvalues = agent.get_value(real_next_obs).reshape(1, -1)
                                nextcostvalues = agent.get_cost_value(real_next_obs).reshape(1, -1)
                            else:
                                nextvalues = values[t + 1]
                                nextcostvalues = cost_values[t + 1]
                            delta = rewards[t] + args.gamma * nextvalues * (1 - terminations[t]) - values[t]
                            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * (1 - dones[t]) * lastgaelam
                            cost_delta = costs[t] + cost_gamma * nextcostvalues * (1 - terminations[t]) - cost_values[t]
                            cost_advantages[t] = lastcostgaelam = cost_delta + cost_gamma * args.gae_lambda * (1 - dones[t]) * lastcostgaelam
                        returns = advantages + values
                        cost_returns = cost_advantages + cost_values
   
                    # Flatten the batch
                    b_obs = observations.reshape((-1,) + envs.single_observation_space.shape)
                    b_logprobs = logprobs.reshape(-1)
                    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                    b_advantages = advantages.reshape(-1)
                    b_cost_advantages = cost_advantages.reshape(-1)
                    b_returns = returns.reshape(-1)
                    b_cost_returns = cost_returns.reshape(-1)
                    b_values = values.reshape(-1)
                    b_cost_values = cost_values.reshape(-1)

                # Dual statistic and multiplier update (shared between async and sync paths)
                if dual_stat == "window8k":
                    dual_jc = costs_q_sum
                elif dual_stat == "episode_mean":
                    dual_jc = float(np.mean(finished_ep_raw_costs)) if finished_ep_raw_costs else float(np.mean(ep_raw_cost_acc))
                else:  # "cost_rate"
                    if finished_ep_rates:
                        dual_jc = float(np.mean(finished_ep_rates))
                    else:
                        dual_jc = float(np.mean((ep_raw_cost_acc + np.maximum(0.0, max_steps - ep_len_acc) * fail_abs) / max_steps))

                if lag_mode == "pid":
                    lag_before = pid_nu
                    err = dual_jc - args.cost_threshold
                    pid_i = max(0.0, pid_i + err)
                    pid_d = 0.0 if pid_prev_jc is None else max(0.0, dual_jc - pid_prev_jc)
                    pid_prev_jc = dual_jc
                    pid_nu = max(0.0, min(args.lag_max, args.lag_kp * err + args.lag_ki * pid_i + args.lag_kd * pid_d))
                    lag_after = pid_nu
                    nu_val = pid_nu
                    multiplier = th.tensor(pid_nu, dtype=th.float32, device=device)
                else:
                    lag_before = float(lag_mul.item())
                    lag_loss = -lag_mul * (th.tensor(dual_jc, dtype=th.float32, device=device) - args.cost_threshold)
                    lag_optim.zero_grad()
                    lag_loss.backward()
                    lag_optim.step()
                    lag_after = float(lag_mul.item())
                    nu_val = float(softplus(lag_mul).item())
                    multiplier = softplus(lag_mul).detach()

                # Optimizing the policy and value network
                b_inds = np.arange(batch_size)
                clipfracs = []
                epochs_run = 0
                for _ in range(args.update_epochs):
                    epochs_run += 1
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions[mb_inds] if continuous_actions else b_actions.long()[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with th.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            #old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        mb_cost_advantages = b_cost_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                            if getattr(args, "norm_cost_adv", False):
                                mb_cost_advantages = (mb_cost_advantages - mb_cost_advantages.mean()) / (mb_cost_advantages.std() + 1e-8)
                            else:
                                mb_cost_advantages -= mb_cost_advantages.mean()

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = th.max(pg_loss1, pg_loss2).mean()
                        pg_r_loss = pg_loss

                        # Cost component
                        if getattr(args, "clip_cost_surr", False):
                            c_loss1 = mb_cost_advantages * ratio
                            c_loss2 = mb_cost_advantages * th.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            surr_cost_advantage = th.max(c_loss1, c_loss2).mean()
                        else:
                            surr_cost_advantage = (mb_cost_advantages * ratio).mean()

                        pg_loss = (pg_loss + multiplier * surr_cost_advantage) / (1 + multiplier)

                        entropy_loss = entropy.mean()
                        pg_loss = pg_loss - args.entropy_coef * entropy_loss

                        actor_optim.zero_grad()
                        pg_loss.backward()
                        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                        actor_optim.step()

                        # Value loss
                        newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                        newcostvalue = agent.get_cost_value(b_obs[mb_inds]).view(-1)
                        if args.clip_vfloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + th.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                            
                            cv_loss_unclipped = (newcostvalue - b_cost_returns[mb_inds]) ** 2
                            cv_clipped = b_cost_values[mb_inds] + th.clamp(
                                newcostvalue - b_cost_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            cv_loss_clipped = (cv_clipped - b_cost_returns[mb_inds]) ** 2
                            cv_loss_max = th.max(cv_loss_unclipped, cv_loss_clipped)
                            cv_loss = 0.5 * cv_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                            cv_loss = 0.5 * ((newcostvalue - b_cost_returns[mb_inds]) ** 2).mean()

                        v_loss *= args.vf_coef
                        cv_loss *= args.vf_coef

                        critic_optim.zero_grad()
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        critic_optim.step()

                        cost_critic_optim.zero_grad()
                        cv_loss.backward()
                        nn.utils.clip_grad_norm_(agent.cost_critic.parameters(), args.max_grad_norm)
                        cost_critic_optim.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                if args.verbose:
                    n_ep = len(finished_ep_costs)
                    ep_cost_mean = float(np.mean(finished_ep_costs)) if n_ep else float("nan")
                    ep_raw_cost_mean = float(np.mean(finished_ep_raw_costs)) if n_ep else float("nan")
                    ep_rate_mean = float(np.mean(finished_ep_rates)) if n_ep else float("nan")
                    ep_len_mean = float(np.mean(finished_ep_lens)) if n_ep else float("nan")
                    print(
                        f"LagrRollout iter={iteration} step={global_step} "
                        f"J_c={dual_jc:.4f} thresh={args.cost_threshold:.4f} "
                        f"lag_before={lag_before:.4f} lag={lag_after:.4f} nu={nu_val:.4f} "
                        f"n_ep={n_ep} ep_cost={ep_cost_mean:.3f} ep_len={ep_len_mean:.1f} "
                        f"kl={float(approx_kl.item()):.5f} clipfrac={float(np.mean(clipfracs)):.4f} "
                        f"pg_r={float(pg_r_loss.item()):.4f} surr_c={float(surr_cost_advantage.item()):.4f} "
                        f"ent={float(entropy_loss.item()):.4f} "
                        f"ep_raw_cost={ep_raw_cost_mean:.3f} ep_rate={ep_rate_mean:.4f} epochs={epochs_run}"
                    )

                # If we reach the node's time limit, we just exit the training loop, save metrics and ckpt
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            # Save the checkpoint and logger data
            pid_state = {"pid_i": pid_i, "pid_prev_jc": pid_prev_jc, "pid_nu": pid_nu}
            ckpt.set_record(args, agent.actor, agent.critic, agent.cost_critic, global_step, actor_optim, critic_optim, cost_critic_optim, cost_threshold, lag_mul, lag_optim, "" if not logger else logger.wb_path, iteration, pid_state=pid_state)
            ckpt.save_obs_rms(envs, args)
            ckpt.save()
            if logger: logger.close()
            envs.close()
