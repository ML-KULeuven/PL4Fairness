from abc import abstractmethod
from pathlib import Path

from typing import Dict, Tuple, Union

import numpy as np
import torch
import pytorch_lightning as pl
from error_parity import RelaxedThresholdOptimizer
from torch import nn
from torch.nn import BCELoss

from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score, binary_auroc, \
	binary_average_precision

from src import ARTIFACTS_PATH
from src.dataset import TabularDataset
from src.metrics import get_all_metrics, probabilistic_statistical_disparity, probabilistic_equalized_odds_disparity, \
	probabilistic_equalized_opportunity_disparity
from src.modules import LightningNN, LightningDPL
from src.training import get_checkpoint_callback, TorchStratifiedKFold, RandomGridSearch, BrierScore
from src.utils import make_standard_from_torch, remove_subsets


class Store:
	def __init__(self):
		self.a = torch.zeros([0], dtype=torch.int32)
		self.y_pred = torch.zeros([0], dtype=torch.float32)
		self.y_true = torch.zeros([0], dtype=torch.int32)

	def add_data(self, a: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor):
		assert a.shape[0] == y_pred.shape[0] and a.shape[0] == y_true.shape[0], "All tensors must have the same length"
		self.a = torch.concatenate((self.a, a.cpu()))
		self.y_pred = torch.concatenate((self.y_pred, y_pred.cpu()))
		self.y_true = torch.concatenate((self.y_true, y_true.cpu()))

	def empty_store(self):
		a = self.a
		y_pred = self.y_pred
		y_true = self.y_true
		self.a = torch.zeros([0], dtype=torch.int32)
		self.y_pred = torch.zeros([0], dtype=torch.float32)
		self.y_true = torch.zeros([0], dtype=torch.int32)
		return a, y_pred, y_true

class Classifier2(nn.Module):
	"""
	Base class for classifiers
	"""

	def __init__(self, name, cfg, best_params=None):
		super().__init__()
		self.name = name
		self.cfg = cfg
		self.eval_metric = cfg.eval_metric
		self.random_state = cfg.seed
		self.val_metric = cfg.stop_metric
		self.best_params = best_params
		self.validation_stopping = cfg['validation_stopping']
		self.stores = []
		self.logic_when_biased = True

	def evaluate_biased(self, mode: bool):
		pass

	@abstractmethod
	def forward(self, *x):
		raise NotImplementedError("This method should be implemented by subclasses")

	def store(self, a: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor, store_idx) -> None:
		"""
		Store the predictions and true labels for later evaluation.
		Input:
			a: tensor of shape [batch_size, 1] with sensitive attribute
			y_pred: tensor of shape [batch_size, 1] with predicted probabilities
			y_true: tensor of shape [batch_size, 1] with true labels
		"""
		while len(self.stores) < store_idx + 1:
			self.stores.append(Store())
		self.stores[store_idx].add_data(a, y_pred, y_true)

	def tune(self, data: Dataset) -> Dict:
		"""
		Tune the classifier by trying different combinations of parameters using 10-fold stratified cross-validation
		Input:
			params: dictionary with parameter names as keys and lists of values to try
			data: training data to use for tuning
		Output:
			Dictionary with the best parameters found during tuning
		"""
		splitter = TorchStratifiedKFold(k=5, shuffle=True, random_state=self.random_state)
		grid_search = RandomGridSearch(m=5, params={"batch_size": [8, 16, 32, 64, 128], "lr": [1e-3, 5e-4, 3e-4, 1e-4, 5e-5]})
		best_val = np.inf
		best_params = None
		# stopped_epochs = []
		for i, params in enumerate(grid_search.get_iter()):
			# for j, (train_data, val_data) in enumerate(splitter.split(data)):
			# 	print('\rTuning parameters... {}/{} - {}/{}'.format(i + 1, grid_search.m, j + 1, splitter.k))
			# 	stopped_epochs.append(self.do_train(train_data, **params, epochs=self.cfg.epochs, val_data=val_data)) # Train model
			# 	self.test(val_data, store_idx=0)	# Test model and store predictions
			train_data, val_data = random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)], generator=torch.Generator().manual_seed(self.cfg.seed))
			stopped_epoch = self.do_train(train_data, **params, epochs=self.cfg.epochs, val_data=val_data)
			self.test(val_data, store_idx=0)
			results = self.evaluate(store_idx=0)	# Evaluate stored predictions and clean storage
			if results[self.eval_metric] < best_val:
				best_val, best_params = results[self.eval_metric], params
				best_params['epochs'] = int(stopped_epoch)
		return best_params

	# def tune2(self, data: Dataset) -> Dict:
	# 	splitter = TorchStratifiedKFold(k=5, shuffle=True, random_state=self.random_state)
	# 	best_global_steps = []
	# 	for j, (train_data, val_data) in enumerate(splitter.split(data)):
	# 		print('\rTuning parameters... {}/{}'.format(j + 1, splitter.k))
	# 		best_global_steps.append(self.do_train(train_data, batch_size=64, lr=3e-4, epochs=80, log=True, test_data=val_data)) # Train model
	# 	# 	self.test(val_data, store_idx=0)	# Test model and store predictions
	# 	# results = self.evaluate(store_idx=0)	# Evaluate stored predictions and clean storage
	# 	return np.median(best_global_steps)

	def fit(self, data: Dataset, val_data: Dataset = None) -> None:
		"""
		Fit the classifier by first tuning parameters and then training the model on the full training data
		Input:
			data: training data to use for tuning
		"""
		if self.best_params is None: # if no params given => tune
			self.best_params = self.tune(data)
		if self.validation_stopping: # if validation stopping => split data into train and validation
			if val_data is None:
				data, val_data = random_split(data, [int(len(data) * 0.9), len(data) - int(len(data) * 0.9)], generator=torch.Generator().manual_seed(self.cfg.seed))
		self.do_train(data, **self.best_params, verbose=True, val_data=val_data)

	def do_train(self, data: Dataset, batch_size, lr, epochs, verbose=False, val_data=None) -> int:
		"""
		Train the classifier on the given dataset.
		Input:
			data: dataset to train the classifier on
			batch_size: size of the batches to use for training
			lr: learning rate to use for training
			epochs: number of epochs to train for
		"""
		dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)#, num_workers=2, persistent_workers=True)
		if val_data is not None:
			val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
			callbacks = [get_checkpoint_callback({"stop_metric": self.val_metric})]
		else:
			val_dataloader = None
			callbacks = []
		trainer = pl.Trainer(default_root_dir=ARTIFACTS_PATH / Path("logs"),
							 accelerator=self.cfg.device,
							 devices=1,
							 max_epochs=epochs,
							 enable_model_summary=verbose,
							 enable_progress_bar=verbose,
							 #log_every_n_steps=1,
							 callbacks=callbacks,
							 val_check_interval=1.0)
		self.evaluate_biased(True)
		self.train(True)
		self.pl_module.lr = lr
		self.pl_module.init_classifier()
		trainer.fit(self.pl_module, dataloader, val_dataloaders=val_dataloader)
		if val_data is not None:
			self.load_checkpoint(trainer.checkpoint_callback.best_model_path)
			return torch.load(trainer.checkpoint_callback.best_model_path)['epoch']
		return epochs

	@abstractmethod
	def load_checkpoint(self, best_model_path) -> None:
		raise NotImplementedError("This method should be implemented by subclasses")

	def test(self, test_data: Dataset, store_idx=0) -> None:
		"""
		Test the classifier on the given dataset and STORE the predictions until evaluate gets called.
		Input:
			test_data: dataset to test the classifier on

		"""
		self.eval()
		with torch.no_grad():
			# a/y_true: torch.tensor(shape=[len(test_data), 1], dtype=torch.int32)
			# y_pred: torch.tensor(shape=[len(test_data), 1], dtype=torch.float32)
			dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2, persistent_workers=True)
			a, y_pred, y_true = self._test(dataloader)
			self.store(a, y_pred, y_true, store_idx)

	def _test(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		a_tot = None
		y_pred_tot = torch.zeros([0], dtype=torch.float32)
		y_true_tot = torch.zeros([0], dtype=torch.int32)
		for a, X, y_true in dataloader:
			y_pred = self.forward(X.to(self.cfg.device), a.to(self.cfg.device)).flatten()
			if len(a.shape) == 1:
				a = a.unsqueeze(1)
			if a_tot is None:
				a_tot = a.cpu()
			else:
				a_tot = torch.concatenate((a_tot, a.cpu()))
			y_pred_tot = torch.concatenate((y_pred_tot, y_pred.cpu()))
			y_true_tot = torch.concatenate((y_true_tot, y_true.cpu()))
		return a_tot, y_pred_tot, y_true_tot

	def enable_postprocessing(self, data: Union[Dataset, Subset]) -> None:
		pass

	def evaluate(self, store_idx) -> Dict:
		assert len(self.stores) > store_idx, "Store index out of range"
		a, y_pred, y_test = self.stores[store_idx].empty_store()

		acc = binary_accuracy(y_pred, y_test)
		f1 = binary_f1_score(y_pred, y_test)
		auc = binary_auroc(y_pred, y_test)
		ap = binary_average_precision(y_pred, y_test)

		brier = BrierScore()(y_pred, y_test)
		ece = BinaryCalibrationError()(y_pred, y_test).item()
		loss = BCELoss()(y_pred, y_test.float()).item()

		if a.shape[1] == 1:
			a = a.squeeze(1)
			psd = probabilistic_statistical_disparity(y_pred, a)
			eqodds = probabilistic_equalized_odds_disparity(y_pred, y_test, a)
			eqopp = probabilistic_equalized_opportunity_disparity(y_pred, y_test, a)
			all_metrics = get_all_metrics(*make_standard_from_torch(a, (y_pred > 0.5).to(torch.int32), y_test))
			metrics = ({k: v for k, v in all_metrics.items() if k in ["Statistical Disparity", "TPR_diff", "FPR_diff"]} |
					   {"Probabilistic Statistical Disparity": psd,
						"EQ ODD Disparity": eqodds,
						"EQ OPP Disparity": eqopp,})
		else:
			metrics = {}
			for i in range(a.shape[1]):
				a_curr = a[:, i]
				psd = probabilistic_statistical_disparity(y_pred, a_curr)
				eqodds = probabilistic_equalized_odds_disparity(y_pred, y_test, a_curr)
				eqopp = probabilistic_equalized_opportunity_disparity(y_pred, y_test, a_curr)
				all_metrics = get_all_metrics(*make_standard_from_torch(a_curr, (y_pred > 0.5).to(torch.int32), y_test))
				metrics |= ({k + f" {i}": v for k, v in all_metrics.items() if
							k in ["Statistical Disparity", "TPR_diff", "FPR_diff"]} |
						   {f"Probabilistic Statistical Disparity {i}": psd,
							f"EQ ODD Disparity {i}": eqodds,
							f"EQ OPP Disparity {i}": eqopp, })

		metrics.update({
			"Accuracy": acc.item(),
			"AUC": auc.item(),
			"AP": ap.item(),
			"F1": f1.item(),
			"Brier Score": brier,
			"Intervention": self.name,
			"ECE": ece,
			"Loss": loss,
		})
		return metrics

class ProbLogClassifier2(Classifier2):
	def __init__(self, cfg, images=False, input_features=5, pos_weight=1, fair_params=None):
		if cfg['tune']:
			best_params = None
		else:
			best_params = {'batch_size': cfg.batch_size, 'lr': cfg.lr, 'epochs': cfg.epochs}
		super().__init__(name="ProbLog Classifier", cfg=cfg, best_params=best_params)
		self.pl_module = LightningDPL(cfg, images=images, input_features=input_features, pos_weight=pos_weight, fair_params=fair_params).to(cfg.device)
		self.fair_params = fair_params
		self.input_features = input_features
		self.images = images
		self.pos_weight = pos_weight
		# do you keep logic when training/testing on biased labels?
		self.logic_when_biased = cfg['as_label'] # (no historical bias OR historical bias as label bias)

	def evaluate_biased(self, mode: bool):
		if self.logic_when_biased:
			self.pl_module.use_logic(mode)
		else:
			self.pl_module.use_logic(not mode)

	def load_checkpoint(self, best_model_path) -> None:
		self.pl_module = LightningDPL.load_from_checkpoint(best_model_path,
														  cfg=self.cfg, images=self.images,
														  input_features=self.input_features,
														   pos_weight=self.pos_weight,
														   fair_params=self.fair_params)

	def forward(self, *x):
		return self.pl_module(*x)

class NNClassifier(Classifier2):
	def __init__(self, name, cfg, images, input_features=5, drop_sensitive=False, pos_weight=1):
		if cfg['tune']:
			best_params = None
		else:
			best_params = {'batch_size': cfg.batch_size, 'lr': cfg.lr, 'epochs': cfg.epochs}
		super().__init__(name, cfg, best_params=best_params)
		self.images = images
		self.input_features = input_features
		self.pos_weight = pos_weight
		self.drop_sensitive = drop_sensitive
		self.pl_module = LightningNN(cfg, images=images, input_features=input_features,
									 drop_sensitive=drop_sensitive, pos_weight=pos_weight).to(cfg.device)

	def load_checkpoint(self, best_model_path) -> None:
		self.pl_module = LightningNN.load_from_checkpoint(best_model_path,
														  cfg=self.cfg, images=self.images,
														  input_features=self.input_features,
														  drop_sensitive=self.drop_sensitive,
														  pos_weight=self.pos_weight)

	def forward(self, X, a):
		return self.pl_module(X)


class PostProcessingWrapper(Classifier2):
	def __init__(self, classifier: Classifier2, constraint='equalized_odds'):
		super().__init__(name=classifier.name + f"-{constraint}", cfg=classifier.cfg)
		self.classifier = classifier
		self.constraint = constraint
		self.use_postprocessing_flag = False

		self.postprocessor = RelaxedThresholdOptimizer(
			predictor=lambda x: self.classifier.forward(torch.tensor(x), torch.zeros(x.shape)).cpu().detach().numpy(), # CANNOT BE USED WITH PROBLOGCLASSIFIER
			constraint=constraint,
		)

	def enable_postprocessing(self, data: Union[Dataset, Subset]) -> None:
		self.use_postprocessing_flag = True
		self.fit_postprocessing(data)

	def fit_postprocessing(self, data):
		data = remove_subsets(data)
		if isinstance(data, TabularDataset):
			X = data.features
			y = data.labels
			a = data.protected_attributes
		else:
			raise NotImplementedError
		self.postprocessor.fit(
			X=X, y=y, group=a
		)

	# overrides Classifier2.store
	def store(self, a: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor, store_idx):
		self.classifier.store(a, y_pred, y_true, store_idx)

	def forward(self, X, a):
		X, a = X.to(self.cfg.device), a.to(self.cfg.device)
		if self.use_postprocessing_flag:
			return torch.tensor(self.postprocessor(X=X.cpu().numpy(), group=a.cpu().numpy()), dtype=torch.float32)
		return self.classifier(X, a)

	def load_checkpoint(self, best_model_path) -> None:
		pass


