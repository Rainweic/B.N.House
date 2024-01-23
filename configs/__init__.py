import json
from types import SimpleNamespace


def read_config(path):
    """json转对象"""
    with open(path) as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
