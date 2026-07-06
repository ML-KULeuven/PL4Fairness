from typing import Union

import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
import torch

from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, Dataset

from src.dataset import TabularDataset


def split_dataset(df: pd.DataFrame, splits: tuple[int], seed: int):
	tmp_df, test_df  = train_test_split(df, test_size=splits[2], random_state=seed)
	train_df, val_df = train_test_split(tmp_df, test_size=splits[1]/(splits[0]+splits[1]), random_state=seed)

	return train_df, val_df, test_df

def make_standard_from_torch(A, y, y_fair):
	# A is a tensor of shape (batch_size, 1) or (batch_size)
	# y is a tensor of shape (batch_size)
	# y_fair is a tensor of shape (batch_size)
	# convert to numpy
	A = A.cpu().detach().numpy()
	y = y.cpu().detach().numpy()
	y_fair = y_fair.cpu().detach().numpy()

	if len(A.shape) == 2:
		dct = {'Y': y, 'Y_fair': y_fair}
		protected_attribute_names = []
		for i in range(A.shape[1]):
			protected_attribute_names += [f"A{i}"]
			dct |= {f'A{i}': A[:, i]}
		df = pd.DataFrame(dct)
	else:
		protected_attribute_names = ["A"]
		df = pd.DataFrame({'A': A, 'Y': y, 'Y_fair': y_fair})
	return make_standard(df, fair=False, protected_attribute_names=protected_attribute_names), make_standard(df, fair=True, protected_attribute_names=protected_attribute_names)

def make_standard(df: pd.DataFrame, fair=True, label_name="Y", protected_attribute_names=["A"]):
	df = df.drop(columns=[name for name in df.columns if "no_hist" in name])
	if fair:
		df = df.drop(columns=["Y"]).rename(columns={"Y_fair": "Y"})
		try:
			df = df.drop(columns=["PR"])
			df = df.drop(columns=[name for name in df.columns if "PQ" in name])
		except KeyError:
			pass
	else:
		df = df.drop(columns=["Y_fair"])
		try:
			df = df.drop(columns=["R"]).rename(columns={"PR": "R"})
			df = df.drop(columns=[name[1:] for name in df.columns if "PQ" in name]).rename(columns={name : name[1:] for name in df.columns if "PQ" in name})
		except KeyError:
			pass
	return StandardDataset(
		df,
		label_name=label_name,
		favorable_classes=[1],
		protected_attribute_names=protected_attribute_names,
		privileged_classes=[[0]],
	)

def get_binary(n, length):
	bin = []
	for i in range(length-1, -1, -1):
		next_bit = n // 2**i
		n -= next_bit * 2**i
		bin.append(next_bit)
	return bin

def np2torch(arr):
	return torch.tensor(arr, dtype=torch.float32)

def df2torch(df):
	return torch.tensor(df.to_numpy(), dtype=torch.float32)

def adapt_for_finetuning(resnet50):
	resnet50.fc = nn.Sequential(
					nn.Flatten(),
					nn.Linear(resnet50.fc.in_features, 1000, bias=True),
					nn.Linear(1000, 1),
					nn.Sigmoid())
	# for p in resnet50.parameters():
	# 	p.requires_grad = False
	# for p in resnet50.fc.parameters():
	# 	p.requires_grad = True
	return resnet50

def clamp_hook(module, minimum=0.0, maximum=1.0):
	with torch.no_grad():
		for param in module.parameters():
			param.clamp_(minimum, maximum)

def get_pos_weight(labels):
	n_pos = labels.sum()
	n_negs = labels.shape[0] - n_pos
	return n_negs / n_pos

def make_unaware(data: StandardDataset):
	"""
	Drop sensitive features from the dataset
	"""
	features = data.feature_names
	sensitive_indices = [features.index(sensitive_f) for sensitive_f in data.protected_attribute_names]
	X = np.delete(data.features, sensitive_indices, axis=1)
	y = data.labels.ravel()
	return X, y

def remove_subsets(X: Union[Subset, Dataset]) -> Dataset:
	idxs_lst = []
	while isinstance(X, Subset):
		idxs_lst = [X.indices] + idxs_lst
		X = X.dataset
	if isinstance(X, TabularDataset):
		return X.take_subset(idxs_lst)
	else:
		raise NotImplementedError()