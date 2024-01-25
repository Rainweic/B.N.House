from gymnasium import Env
from gymnasium import spaces


class TradeEnv(Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(self) -> None:
        super().__init__()

        # Action dict:
        # 0: 做空  1: 做多  2: 保持当前仓位
        self.action_space = spaces.Discrete()
        self._action_to_trade = {
            0: ...,
            1: ...,
            2: ...,
        }

        self.observation_space = ...

    def reset(self, *, seed, options):
        return super().reset(seed=seed, options=options)

    def step(self, action):
        return super().step(action)
