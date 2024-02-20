import json
from types import SimpleNamespace


def read_config(path):
    """json转对象"""
    with open(path) as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))


# test
if __name__ == '__main__':
    config = read_config('/Users/new/B.N.House/configs/dqn_trade.json')
    print(config.model_kwargs.__dict__)
