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
from tianshou.policy import BasePolicy
from tianshou.data.batch import Batch

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
        eval_envs = SubprocVectorEnv([lambda: gym.make(config.env_name) for _ in range(config.test_num)])
    else:
        eval_envs = gym.make(config.env_name)

    # set seed
    if hasattr(config, "random_seed"):
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        train_envs.seed(config.random_seed)
        eval_envs.seed(config.random_seed)

    # Model
    model = MODEL_ZOO[config.model](config=config)

    # GPU or CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.to("cuda")

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
        VectorReplayBuffer(config.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, eval_envs, exploration_noise=True)

    # Log
    log_path = os.path.join("./logs", config.env_name, config.model)
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

    # test
    if stop_fn(result.get("best_reward")):

        if hasattr(config, "save_model_name"):
            save_name = config.save_model_name
        else:
            save_name = "policy.pth"
        policy.load_state_dict(torch.load(os.path.join(log_path, save_name)))

        test_env = gym.make(config.env_name)
        obs, info = test_env.reset()

        act = policy(Batch(obs=obs[np.newaxis, :], info={})).act.item()
        act = policy.map_action(act)

        while True:
            obs, reward, terminated, truncated, info = test_env.step(act)

            if terminated or truncated:
                break

            act = policy(Batch(obs=obs[np.newaxis, :], info={})).act.item()
            act = policy.map_action(act)

        test_env.close()


if __name__ == "__main__":
    args = get_args()
    train(args)
