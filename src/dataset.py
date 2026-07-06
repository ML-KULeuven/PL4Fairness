import os
from pathlib import Path

import PIL
import numpy as np
import torch

from torch.utils.data import Dataset, TensorDataset, Subset
from torchvision import transforms

class ImageSensitiveDataset(Dataset):
	def __init__(self, index, X, labels, device='cpu', root=Path("res/data/celeba")):
		self.index = index
		self.sensitive = torch.tensor(X, dtype=torch.int32)
		self.labels = torch.tensor(labels, dtype=torch.int32).flatten()
		self.device = device

		self.root = root
		self.base_folder = "celeba"

		self.transform = transforms.Compose([
				transforms.CenterCrop(178),          # keep face region
				transforms.Resize(128),              # make every image 128×128
				transforms.ToTensor(),               # HWC 0‑255 → CHW 0‑1
				transforms.Normalize(                # ImageNet‑style stats
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225]),
			])

	def __len__(self):
		return len(self.sensitive)

	def __getitem__(self, idx):
		X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.index[idx]))
		X = self.transform(X)
		a = self.sensitive[idx]
		label = self.labels[idx]

		return a, X, label

class TabularDataset(Dataset):
	def __init__(self, X: torch.Tensor, y: torch.Tensor, sensitive_idx: int, feature_names=None):
		a = X[:, sensitive_idx].to(torch.int32)
		self.feature_names = feature_names
		self.sensitive_idx = sensitive_idx
		self.features = X.numpy().copy()
		self.protected_attributes = a.numpy().copy()
		self.labels = y.numpy().copy()
		self.data = TensorDataset(a, X, y)

	def __getitem__(self, idx):
		return self.data.__getitem__(idx)

	def copy(self):
		return TabularDataset(
			X=torch.tensor(self.features, dtype=torch.float32),
			y=torch.tensor(self.labels, dtype=torch.float32),
			sensitive_idx=self.sensitive_idx,
			feature_names=self.feature_names
		)

	def _take_subset(self, dataset, idxs):
		dataset.features = dataset.features[idxs]
		dataset.protected_attributes = dataset.protected_attributes[idxs]
		dataset.labels = dataset.labels[idxs]
		dataset.data = Subset(dataset.data, idxs)
		return dataset

	def take_subset(self, idxs_lst):
		subset_dataset = self.copy()
		for idxs in idxs_lst:
			subset_dataset = self._take_subset(subset_dataset, idxs)
		return subset_dataset

	def set_labels(self, y: np.ndarray):
		self.labels = y
		self.data = TensorDataset(
			torch.tensor(self.protected_attributes, dtype=torch.int32),
			torch.tensor(self.features, dtype=torch.float32),
			torch.tensor(y, dtype=torch.int32)
		)

	def __len__(self):
		return len(self.data)