import torch
import numpy as np
from typing import Any, Dict, Tuple, Union
from torch import nn


class RainBow4Trade(nn.Module):

    def __init__(self, feature_size, timestep, output_size, gru_num_layers=2,
                 softmax=False, device='cpu'):
        super(RainBow4Trade, self).__init__()

        self.gru = nn.GRU(feature_size, timestep, num_layers=gru_num_layers, batch_first=True)
        self.fc = nn.Linear(timestep, output_size)
        self.softmax = softmax
        
        self.device = device

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        output, _ = self.gru(obs)
        output = self.fc(output[:, -1, :])

        if self.softmax:
            output = torch.softmax(output, dim=1)
        return output, state


# test the model
if __name__ == '__main__':

    bs = 1
    feature_size = 22
    timestep = 5
    output_size = 2

    model = RainBow4Trade(feature_size=feature_size, timestep=timestep, output_size=output_size)
    obs = torch.randn(bs, timestep, feature_size)
    output, _ = model(obs)
    print(output.shape)
    print(output)
