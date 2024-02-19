import random
import warnings
import numpy as np
import pandas as pd
import gymnasium as gym

from enum import Enum
from gymnasium import Env
from gymnasium import spaces
from gymnasium.envs.registration import register
from datetime import datetime
from mongoengine import connect

from env.core import calc_reward
from quant_trading.models import PickleDbTicks


warnings.filterwarnings("ignore")


__all__ = ["TradeEnv", "SEL_COLS"]


SEL_COLS = ['LastPrice', 'LastVolume',
            'AskPrice1', 'AskVolume1', 'BidPrice1', 'BidVolume1',
            'AskPrice2', 'AskVolume2', 'BidPrice2', 'BidVolume2',
            'AskPrice3', 'AskVolume3', 'BidPrice3', 'BidVolume3',
            'AskPrice4', 'AskVolume4', 'BidPrice4', 'BidVolume4',
            'AskPrice5', 'AskVolume5', 'BidPrice5', 'BidVolume5'
            ]
INF = 1e10


class Actions(Enum):
    Sell = 0
    Buy = 1


class TradeEnv(Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cat, timestep, start_time, end_time, stop_loss_th_init, multiplier,
                 time_type='random', URI_ticks='mongodb://ticks:11112222@mongodb:27017/ticks',
                 use_fake_data=False):
        super().__init__()

        # 连接数据库
        # URI_ticks = 'mongodb://ticks:11112222@mongodb:27017/ticks'
        # URI_kline = 'mongodb://127.0.0.1:6007/kline'
        connect(host=URI_ticks,  alias='ticks')

        self.cat = cat
        self.timestep = timestep
        self.start_time = start_time
        self.end_time = end_time
        self.stop_loss_th_init = stop_loss_th_init
        self.multiplier = multiplier
        self.time_type = time_type
        self.use_fake_data = use_fake_data

        # 加载数据
        self.df = self._load_data()

        # 定义space
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_shape = (timestep, len(self.df.columns))
        self.observation_space = spaces.Box(
            low=-INF, high=INF, shape=self.observation_shape, dtype=np.float32
        )

    def _get_observation(self):
        return self.df.iloc[self.sel_point-self.timestep:self.sel_point][SEL_COLS]

    def _get_info(self):
        return dict(
            cat=self.cat,
            timestep=self.timestep,
            start_time=self.start_time,
            end_time=self.end_time,
            stop_loss_th_init=self.stop_loss_th_init,
            multiplier=self.multiplier,
            total_reward=self.total_reward,
            mean_reward=self.mean_reward,
            max_reward=self.max_reward
        )

    def _load_data(self):

        # 根据日期选择数据
        sel_days = [datetime.strftime(x, '%Y%m%d') for x in
                    list(pd.date_range(start=self.start_time, end=self.end_time))]

        if self.use_fake_data:
            # 随机生成数据 仅用于测试
            df = pd.DataFrame(np.random.rand(len(sel_days), len(SEL_COLS)), columns=SEL_COLS)
            return df

        # 加载数据
        ticks = PickleDbTicks(dict(category=self.cat, subID='9999', day__in=sel_days), main_cls='')
        df = ticks.load_ticks()
        df = df.reset_index(drop=True)

        # night数据是分开两天存储的
        try:
            _df = df[(df.day == sel_days[1]) & (df.time_type != 'night')]
            df = df.iloc[:_df.index[0]]
        except:
            print(f'sth went wrong.')

        # 随机选择上午、下午、晚上
        if self.time_type == "random":
            t_type = random.choice(['night', 'am_pm'])
        else:
            t_type = self.time_type
        if t_type == 'am_pm':
            df = df[(df.time_type == 'fam') | (df.time_type == 'bam') | (df.time_type == 'pm')]
        else:
            df = df[df.time_type == t_type]

        return df

    def reset(self):

        # 初始化参数
        self.truncated = False
        self.sel_point = self.timestep
        self.total_reward = 0
        self.mean_reward = 0
        self.max_reward = 0
        self.step_times = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):

        if self.sel_point >= self.df.shape[0]:
            self.truncated = True

        if not self.truncated:
            observation = self._get_observation()
            df_op = self.df.iloc[self.sel_point:]
            if action == Actions.Buy.value:
                reward = calc_reward(df_op, 'buy', self.stop_loss_th_init, self.multiplier)
            elif action == Actions.Sell.value:
                reward = calc_reward(df_op, 'sell', self.stop_loss_th_init, self.multiplier)

            if reward > self.max_reward:
                self.max_reward = reward
            self.step_times += 1
            self.sel_point += 1
            self.total_reward += reward
            self.mean_reward = self.total_reward / self.step_times
        else:
            observation = None
            reward = 0

        info = self._get_info()

        return observation, reward, False, self.truncated, info


register(id="trade", entry_point="env.trade_env:TradeEnv")


# test
if __name__ == "__main__":
    env = gym.make(id='trade', cat='AU', timestep=5, start_time='20220524', end_time='20221230',
                   stop_loss_th_init=0.0025, multiplier=500, use_fake_data=True)
    env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"obs: {observation}\n reward: {reward}")

        if terminated or truncated:
            observation, info = env.reset()
