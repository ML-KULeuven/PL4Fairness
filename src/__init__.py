import itertools
from collections.abc import Iterator
from pathlib import Path

import random
from typing import Dict

import numpy as np
import yaml
import torch

DATA_ROOT = Path("res/data")
ARTIFACTS_PATH = Path("res/artifacts")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"ConfigDict has no attribute {name}")

    def save(self, file_name: str | Path):
        with open(file_name, "w+") as f:
            yaml.dump(dict(self), f)

    def copy(self):
        return ConfigDict(self)

    def add(self, other: Dict):
        for key in other:
            if key in self:
                raise KeyError(f"Key {key} already exists in ConfigDict.")
            else:
                self[key] = other[key]

class GridSearch(Iterator):
    def __init__(self, cfg, grid_params):
        self.cfg = cfg
        self.grid_params = grid_params # first argument is outermost loop
        self.do_asserts()
        self.value_product = itertools.product(*map(self.cfg.get, self.grid_params))

    def do_asserts(self):
        for param in self.grid_params:
            assert isinstance(self.cfg[param], list), f"Parameter {param} should be a list in the config file."

    def __next__(self) -> ConfigDict:
        vs = next(self.value_product)
        tmp_cfg = self.cfg.copy()
        for i,p in enumerate(self.grid_params):
            tmp_cfg[p] = vs[i]
        return tmp_cfg

def load_config(cfg) -> ConfigDict:
    if isinstance(cfg, dict):
        return ConfigDict(cfg)
    cfg = str(cfg)
    with open(cfg) as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))