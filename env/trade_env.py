import random
import numpy as np
import gymnasium as gym

from enum import Enum
from gymnasium import Env
from gymnasium import spaces
from gymnasium.envs.registration import register
from mongoengine import *
from models import *

from env.core import calc_reward


DROP_COLS = [
        'InstrumentID', 'MarketID', 'mainID',
        # 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5',
        # 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
        'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
        'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
    ]


class Actions(Enum):
    Sell = 0
    Buy = 1


class TradeEnv(Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cat, feed_data_length, start_time, end_time, stop_loss_th_init, multiplier,
                 time_type='random'):
        super().__init__()

        # 连接数据库
        URI_ticks = 'mongodb://ticks:11112222@mongodb:27017/ticks'
        # URI_kline = 'mongodb://127.0.0.1:6007/kline'
        connect(host=URI_ticks,  alias='ticks')

        self.cat = cat
        self.feed_data_length = feed_data_length
        self.start_time = start_time
        self.end_time = end_time
        self.stop_loss_th_init = stop_loss_th_init
        self.multiplier = multiplier
        self.time_type = time_type

        # 加载数据
        self.df = self._load_data()

        # 定义space
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(
            low=-INF, high=INF, shape=self.observation_shape, dtype=np.float32
        )
        self.observation_shape = (feed_data_length, len(self.df.columns))

    def _get_observation(self):
        return self.df.iloc[self.sel_point-self.feed_data_length:self.sel_point]

    def _get_info(self):
        return dict(
            cat=self.cat,
            feed_data_length=self.feed_data_length,
            start_time=self.start_time,
            end_time=self.end_time,
            stop_loss_th_init=self.stop_loss_th_init,
            multiplier=self.multiplier,
            total_reward=self.total_reward,
            mean_reward=self.mean_reward,
            max_reward=self.max_reward
        )

    def _load_data(self):

        # 加载日期
        ticks = PickleDbTicks(dict(category=self.cat, subID='9999'), main_cls='')
        all_days = ticks.ticks.distinct('day')
        all_days = sorted(all_days)
        start_time_idx = all_days.index(self.start_time)
        end_time_idx = all_days.index(self.end_time)
        sel_days = all_days[start_time_idx:end_time_idx]

        # 加载数据
        ticks = PickleDbTicks(dict(category=self.cat, subID='9999', day__in=sel_days), main_cls='')
        df = ticks.load_ticks()
        df = df.drop(DROP_COLS, axis=1)
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
        self.sel_point = self.feed_data_length
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


register(id="trade", entry_point="envs.trade_env:TradeEnv")


# test
if __name__ == "__main__":
    env = gym.make(id='trade', cat='AU', feed_data_length=5, start_time='20180524', end_time='20221230',
                   stop_loss_th_init=0.0025, multiplier=500)
    env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
