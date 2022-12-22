# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import mindspore as ms
from mindspore import ops, nn, ms_function
from mindspore.common.initializer import initializer, Constant, Orthogonal
from mindspore.nn.probability.distribution import Categorical
from tensorboardX import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--cuda", default=False, action='store_true',
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", default=False, action='store_true',
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", default=False, action='store_true',
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer: nn.Cell, std=np.sqrt(2), bias_const=0.0):
    layer.weight.set_data(initializer(Orthogonal(std), layer.weight.shape, layer.weight.dtype))
    layer.bias.set_data(initializer(Constant(bias_const), layer.bias.shape, layer.bias.dtype))

    return layer


class Agent(nn.Cell):
    def __init__(self, envs):
        super().__init__()
        input_size = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.SequentialCell(
            layer_init(nn.Dense(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Dense(64, 64)),
            nn.Tanh(),
            layer_init(nn.Dense(64, 1), std=1.0),
        )
        self.actor = nn.SequentialCell(
            layer_init(nn.Dense(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Dense(64, 64)),
            nn.Tanh(),
            layer_init(nn.Dense(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(ops.softmax(logits, axis=-1))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.cuda:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU", device_id=0)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    ms.set_seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs)
    optimizer = nn.Adam(agent.trainable_params(), learning_rate=args.learning_rate, eps=1e-5)
    optimizer.learning_rate.set_data(0.025)

    # ALGO Logic: Storage setup
    obs = ms.numpy.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = ms.numpy.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = ms.numpy.zeros((args.num_steps, args.num_envs))
    rewards = ms.numpy.zeros((args.num_steps, args.num_envs))
    dones = ms.numpy.zeros((args.num_steps, args.num_envs))
    values = ms.numpy.zeros((args.num_steps, args.num_envs))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = ms.Tensor(envs.reset())
    next_done = ms.numpy.zeros(args.num_envs)
    num_updates = args.total_timesteps // args.batch_size
    norm_adv, clip_coef, clip_vloss, ent_coef, vf_coef, max_grad_norm = args.norm_adv, args.clip_coef, \
        args.clip_vloss, args.ent_coef, \
        args.vf_coef, args.max_grad_norm


    def forward_fn(b_observations, b_actions, b_logprobs, b_returns, b_advantages, b_values):
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_observations, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        mb_advantages = b_advantages
        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * ms.numpy.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = ops.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if clip_vloss:
            v_loss_unclipped = (newvalue - b_returns) ** 2
            v_clipped = b_values + ms.numpy.clip(
                newvalue - b_values,
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = ops.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

        return loss, v_loss, pg_loss, entropy_loss, logratio, ratio


    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


    # @ms_function
    def train_step(b_observations, b_actions, b_logprobs, b_returns, b_advantages, b_values):
        (loss, v_loss, pg_loss, entropy_loss, logratio, ratio), grads = grad_fn(b_observations, b_actions, b_logprobs,
                                                                                b_returns, b_advantages, b_values)
        grads = ops.clip_by_global_norm(grads, clip_norm=max_grad_norm)
        optimizer(grads)
        return v_loss, pg_loss, entropy_loss, logratio, ratio


    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.learning_rate.set_data(lrnow)

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.asnumpy())
            rewards[step] = ms.Tensor(reward).view((-1,))
            next_obs, next_done = ms.Tensor(next_obs), ms.Tensor(done)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        next_value = agent.get_value(next_obs).reshape((-1,))
        advantages = ops.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end].tolist()
                v_loss, pg_loss, entropy_loss, logratio, ratio = train_step(b_obs[mb_inds],
                                                                            ops.cast(b_actions, ms.int32)[mb_inds],
                                                                            b_logprobs[mb_inds],
                                                                            b_returns[mb_inds],
                                                                            b_advantages[mb_inds],
                                                                            b_values[mb_inds]
                                                                            )

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [ops.cast((ratio - 1.0).abs() > args.clip_coef, ms.float32).mean().asnumpy()]

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.asnumpy(), b_returns.asnumpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", lrnow, global_step)
        writer.add_scalar("losses/value_loss", v_loss.asnumpy(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.asnumpy(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.asnumpy(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.asnumpy(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.asnumpy(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
