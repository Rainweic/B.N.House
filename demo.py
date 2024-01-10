import gymnasium as gym

# 创建环境
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()


while True:

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()


env.close()
