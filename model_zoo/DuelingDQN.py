import torch
from tianshou.utils.net.common import Net


class DuelingDQN(Net):

    def __init__(self, config):

        Q_param = {"hidden_sizes": config.dueling_q_hidden_sizes}
        V_param = {"hidden_sizes": config.dueling_v_hidden_sizes}

        device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(
            device=device,
            state_shape=config.state_shape,
            action_shape=config.action_shape,
            hidden_sizes=config.hidden_sizes,
            dueling_param=(Q_param, V_param)
        )
