from gymnasium import Env


class TradeEnv(Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(self) -> None:
        super().__init__()

        self.action_space = ...
