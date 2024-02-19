import torch
import numpy as np
from typing import Any, Dict, Tuple, Union
from torch import nn


class RainBow4Trade(nn.Module):

    def __init__(self, feature_size, timestep, output_size, gru_num_layers=2,
                 softmax=False):
        super(RainBow4Trade, self).__init__()

        self.gru = nn.GRU(feature_size, timestep, num_layers=gru_num_layers, batch_first=True)
        self.fc = nn.Linear(timestep, output_size)

        self.softmax = softmax

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        output, _ = self.gru(obs)
        output = self.fc(output[:, -1, :])

        if self.softmax:
            output = torch.softmax(output, dim=1)
        return output, state


# test the model
if __name__ == '__main__':
    model = RainBow4Trade(10, 5, 1)
    obs = torch.randn(3, 5, 10)
    output, _ = model(obs)
    print(output.shape)
    print(output)
