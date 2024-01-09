import gymnasium as gym

# 创建环境
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

# 开始游戏
while True:
    action = env.action_space.sample()  # 随机选择动作
    observation, reward, done, info = env.step(action)  # 执行动作并获取反馈
    if done:  # 游戏结束
        break
    
    # 显示游戏画面
    env.render()
    

# 关闭环境
env.close()
