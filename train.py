import os

import gymnasium as gym
import numpy as np
import torch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from tianshou.policy import BasePolicy

from common import get_args, OPTIMIZER, LOG_PATH
from configs import read_config
from model_zoo import MODEL_ZOO
from policy_zoo import POLICY_ZOO
from test import test
from env.trade_env import *


def train(args):

    config = read_config(args.c)

    # set state_shape、action_shape
    env_kwargs = config.env_kwargs.__dict__
    print(f"env_kwargs:\n{env_kwargs}")
    
    tmp_env = gym.make(**env_kwargs)
    config.state_shape = tmp_env.observation_space.shape or tmp_env.observation_space.n
    config.action_shape = tmp_env.action_space.shape or tmp_env.action_space.n

    # make train/test env
    if hasattr(config, "training_num"):
        train_envs = DummyVectorEnv([lambda: gym.make(**env_kwargs) for _ in range(config.training_num)])
    else:
        train_envs = gym.make(**env_kwargs)

    if hasattr(config, "test_num"):
        eval_envs = SubprocVectorEnv([lambda: gym.make(**env_kwargs) for _ in range(config.test_num)])
    else:
        eval_envs = gym.make(**env_kwargs)

    # set seed
    if hasattr(config, "random_seed"):
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        train_envs.seed(config.random_seed)
        eval_envs.seed(config.random_seed)

    # Model
    if hasattr(config, "model_kwargs"):
        if config.model_kwargs.feature_size == "state_shape":
            config.model_kwargs.feature_size = config.state_shape[-1]
        if config.model_kwargs.output_size == "action_shape":
            config.model_kwargs.output_size = config.action_shape
        model_kwargs = config.model_kwargs.__dict__
        print(f"model_kwargs:\n{model_kwargs}")
        model = MODEL_ZOO[config.model](**model_kwargs)
    else:
        model = MODEL_ZOO[config.model](config=config)

    # GPU or CPU
    device = config.model_kwargs.device
    model.to(device)

    # Optimizer
    optim = OPTIMIZER[config.optimizer](model.parameters(), lr=config.lr)

    # Policy
    policy: BasePolicy = POLICY_ZOO[config.policy](
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
        VectorReplayBuffer(config.buffer_size, len(train_envs) if isinstance(train_envs, SubprocVectorEnv) else 1),
        exploration_noise=True,
    )
    test_collector = Collector(policy, eval_envs, exploration_noise=True)

    # Log
    log_path = os.path.join(LOG_PATH, config.env_kwargs.id, config.model)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        if hasattr(config, "save_model_name"):
            save_name = config.save_model_name
        else:
            save_name = "policy.pth"
        torch.save(policy.state_dict(), os.path.join(log_path, save_name))

    def stop_fn(mean_rewards):
        return mean_rewards >= tmp_env.spec.reward_threshold

    def train_fn(epoch, env_step):  # exp decay
        eps = max(config.eps_train * (1 - 5e-6) ** env_step, config.eps_test)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(config.eps_test)

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config.epoch,
        step_per_epoch=config.step_per_epoch,
        step_per_collect=config.step_per_collect,
        episode_per_test=config.test_num,
        batch_size=config.batch_size,
        update_per_step=config.update_per_step,
        stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    print(result)

    # test
    test(args)


if __name__ == "__main__":
    args = get_args()
    train(args)
