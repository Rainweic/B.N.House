import os
import argparse

import gymnasium as gym
import numpy as np
import torch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer

from configs import read_config
from model_zoo import MODEL_ZOO
from policy_zoo import POLICY_ZOO


OPTIMIZER = {
    "Adam": torch.optim.Adam
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="path of config file", default="./configs/dqn_lunarlander.json")
    return parser.parse_args()


def train(args):

    config = read_config(args.c)

    # set state_shape、action_shape
    tmp_env = gym.make(config.env_name)
    config.state_shape = tmp_env.observation_space.shape or tmp_env.observation_space.n
    config.action_shape = tmp_env.action_space.shape or tmp_env.action_space.n

    # make train/test env
    if hasattr(config, "training_num"):
        train_envs = DummyVectorEnv([lambda: gym.make(config.env_name) for _ in range(config.training_num)])
    else:
        train_envs = gym.make(config.env_name)

    if hasattr(config, "test_num"):
        test_envs = SubprocVectorEnv([lambda: gym.make(config.env_name) for _ in range(config.test_num)])
    else:
        test_envs = gym.make(config.env_name)

    # set seed
    if hasattr(config, "random_seed"):
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        train_envs.seed(config.random_seed)
        test_envs.seed(config.random_seed)

    # Model
    model = MODEL_ZOO[config.model](config=config)

    # GPU or CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.to("gpu")

    # Optimizer
    optim = OPTIMIZER[config.optimizer](model.parameters(), lr=config.lr)

    # Policy
    policy = POLICY_ZOO[config.policy](
        model=model,
        optim=optim,
        action_space=tmp_env.action_space,
        discount_factor=config.gamma,
        estimation_step=config.n_step,
        target_update_freq=config.target_update_freq,
    )

    # Collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(config.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # Log
    log_path = os.path.join("./logs", config.env_name, config.model)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= tmp_env.spec.reward_threshold

    def train_fn(epoch, env_step):  # exp decay
        eps = max(args.eps_train * (1 - 5e-6) ** env_step, args.eps_test)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    assert stop_fn(result.best_reward)


if __name__ == "__main__":
    args = get_args()
    train(args)
