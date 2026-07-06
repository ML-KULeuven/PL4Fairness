import random
from itertools import product
from typing import Iterator, Tuple, List, Dict

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import Dataset, Subset


def _metric_mode(stop_metric) -> str:
    if "auroc" in stop_metric or 'pr' in stop_metric or 'acc' in stop_metric:
        return 'max'
    return 'min'

def get_early_stopping_callback(cfg) -> EarlyStopping:
    return EarlyStopping(monitor=cfg["stop_metric"], patience=cfg["patience"], verbose=cfg["verbose"],
                         mode=_metric_mode(cfg["stop_metric"]))

def get_checkpoint_callback(cfg) -> ModelCheckpoint:
    return ModelCheckpoint(save_top_k=1, monitor=cfg['stop_metric'], mode=_metric_mode(cfg['stop_metric']))

class BrierScore(nn.Module):
    def __init__(self):
        super(BrierScore, self).__init__()

    def forward(self, inputs, targets):
        return brier_score_loss(targets, inputs)

class BalancedBCELoss(nn.Module):
    def __init__(self, pos_weight=1):
        super(BalancedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        weights = torch.where(targets == 1, self.pos_weight, 1.0)
        intermediate_losses = self.bce_loss(inputs, targets.to(torch.float32))
        return torch.mean(weights * intermediate_losses)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets.to(torch.float32))
        pt = torch.exp(-bce_loss)
        alpha = torch.where(targets == 1, self.alpha, 1.0)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class TorchStratifiedKFold:
    def __init__(self, k, shuffle=False, random_state=None):
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state
        self.splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    def split(self, data: Dataset) -> Iterator[Tuple[Subset, Subset]]:
        labels = [y for _, _, y in data]
        for train_idxs, test_idxs in self.splitter.split(labels, labels):
            yield Subset(data, train_idxs), Subset(data, test_idxs)

    def split2(self, datasets: List[Dataset]) -> Iterator[Tuple[List[Subset], List[Subset]]]:
        labels = [y for _, _, y in datasets[0]] # stratify according to the first dataset's labels
        for train_idxs, test_idxs in self.splitter.split(labels, labels):
            yield [Subset(data, train_idxs) for data in datasets], [Subset(data, test_idxs) for data in datasets]

class RandomGridSearch:
    def __init__(self, m: int, params: Dict[str, List]):
        self.m = m
        self.params = params

    def get_iter(self) -> Iterator:
        names = list(self.params.keys())
        all_choices = list(product(*self.params.values()))
        choices = random.sample(all_choices, self.m)
        for values in choices:
            yield {n:v for n, v in zip(names, values)}