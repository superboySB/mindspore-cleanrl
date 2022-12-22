# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import copy
import os
import random
import time

import sys
import gym
import math
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import mindspore as ms
from mindspore import ops, nn, ms_function
from mindspore.common.initializer import Uniform, HeUniform, initializer
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())
from cleanrl_utils.utils import sync_weight


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
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
                        help="the frequency of training")
    args = parser.parse_args()
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


def layer_init(layer: nn.Cell, std=math.sqrt(5)):
    layer.weight.set_data(initializer(HeUniform(negative_slope=std), layer.weight.shape, layer.weight.dtype))
    layer.bias.set_data(
        initializer(Uniform(scale=1 / math.sqrt(layer.in_channels)), layer.bias.shape, layer.bias.dtype))

    return layer


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Cell):
    def __init__(self, env):
        super().__init__()
        input_size = int(np.array(env.single_observation_space.shape).prod())
        self.network = nn.SequentialCell(
            layer_init(nn.Dense(input_size, 120)),
            nn.ReLU(),
            layer_init(nn.Dense(120, 84)),
            nn.ReLU(),
            layer_init(nn.Dense(84, env.single_action_space.n)),
        )

    def construct(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs)
    target_network = copy.deepcopy(q_network)
    q_network_list = nn.CellList()
    q_network_list.append(q_network)
    q_network_list.append(target_network)
    optimizer = nn.Adam(q_network.trainable_params(), learning_rate=args.learning_rate)
    sync_weight(model=q_network, model_old=target_network)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        handle_timeout_termination=True,
    )
    gamma = args.gamma

    def forward_fn(observations: ms.Tensor, actions: ms.Tensor, rewards: ms.Tensor,
                   next_observations: ms.Tensor, dones: ms.Tensor):
        target_max = target_network(next_observations).max(axis=1)
        td_target = rewards.flatten() + gamma * target_max * (1 - dones.flatten())
        old_val = ops.gather_elements(q_network(observations), dim=1, index=actions).squeeze()
        loss = nn.MSELoss("mean")(ops.stop_gradient(td_target), old_val)
        return loss, old_val

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    @ms_function
    def train_step(observations: ms.Tensor, actions: ms.Tensor, rewards: ms.Tensor,
                   next_observations: ms.Tensor, dones: ms.Tensor):
        (loss, old_val), grads = grad_fn(observations, actions, rewards, next_observations, dones)
        optimizer(grads)
        return loss, old_val


    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(ms.Tensor(obs))
            actions = q_values.argmax(axis=1).asnumpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                observations = ms.Tensor(data.observations.numpy(), dtype=ms.float32)
                actions = ms.Tensor(data.actions.numpy(), dtype=ms.int32)
                rewards = ms.Tensor(data.rewards.numpy(), dtype=ms.float32)
                next_observations = ms.Tensor(data.next_observations.numpy(), dtype=ms.float32)
                dones = ms.Tensor(data.dones.numpy(), dtype=ms.float32)

                loss, old_val = train_step(observations, actions, rewards, next_observations, dones)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.asnumpy(), global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().asnumpy(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update the target network
            if global_step % args.target_network_frequency == 0:
                a = sync_weight(model=q_network, model_old=target_network)

    envs.close()
    writer.close()
