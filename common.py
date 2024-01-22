import torch
import argparse


OPTIMIZER = {
    "Adam": torch.optim.Adam
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="path of config file", default="./configs/dqn_lunarlander.json")
    return parser.parse_args()