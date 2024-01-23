import os
import torch
import numpy as np
import gymnasium as gym

from common import get_args
from configs import read_config
from tianshou.policy import BasePolicy
from tianshou.data import Batch

from common import OPTIMIZER, LOG_PATH
from model_zoo import MODEL_ZOO
from policy_zoo import POLICY_ZOO


def test(args):

    config = read_config(args.c)

    test_env = gym.make(config.env_name, render_mode="human")

    config.state_shape = test_env.observation_space.shape or test_env.observation_space.n
    config.action_shape = test_env.action_space.shape or test_env.action_space.n

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
        action_space=test_env.action_space,
        discount_factor=config.gamma,
        estimation_step=config.n_step,
        target_update_freq=config.target_update_freq,
    )

    if hasattr(config, "save_model_name"):
        save_name = config.save_model_name
    else:
        save_name = "policy.pth"
    log_path = os.path.join(LOG_PATH, config.env_name, config.model)

    policy.load_state_dict(torch.load(
        os.path.join(log_path, save_name),
        map_location=torch.device('cuda' if use_cuda else 'cpu')
    ))

    policy.eval()

    if hasattr(policy, "eps_test"):
        policy.set_eps(config.eps_test)

    obs, info = test_env.reset()

    while True:

        act = policy(Batch(obs=obs[np.newaxis, :], info={})).act.item()
        act = policy.map_action(act)

        obs, reward, terminated, truncated, info = test_env.step(act)
        print(f"act: {act}, reward: {reward}")

        if terminated or truncated:
            print("end")
            observation, info = test_env.reset()
            break

    test_env.close()


if __name__ == "__main__":
    args = get_args()
    test(args)
