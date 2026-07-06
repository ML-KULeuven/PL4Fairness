from typing import Tuple

import numpy as np
import pandas as pd
import torch

from aif360.datasets import StandardDataset
from torch.utils.data import Subset

from src import ConfigDict
from src.classifier import Classifier2, ProbLogClassifier2, NNClassifier, PostProcessingWrapper
from src.dataset import ImageSensitiveDataset, TabularDataset
from src.massaging import Massaging
from src.training import TorchStratifiedKFold
from src.utils import get_pos_weight


class Tester:
	def __init__(self, data_fair: StandardDataset, data_biased: StandardDataset,
				 val_fair: StandardDataset = None, val_biased: StandardDataset = None,
				 test_fair: StandardDataset = None, test_biased: StandardDataset = None,
				 cfg: ConfigDict=None, images=False,
				 data_drift: StandardDataset=None):
		self.data_fair = data_fair
		self.data_biased = data_biased
		self.data_drift = data_drift
		if cfg['use_pos_weight']:
			self.fair_pos_weight = get_pos_weight(data_fair.labels)
			self.biased_pos_weight = get_pos_weight(data_biased.labels)
		else:
			self.fair_pos_weight = 1
			self.biased_pos_weight = 1
		self.nb_features = len(data_fair.feature_names)
		self.cfg = cfg
		self.images = images
		self.baseline_biased_results = []
		self.baseline_fair_results = []
		self.problog_biased_results = []
		self.problog_fair_results = []
		pos_fair = np.sum(self.data_fair.labels)
		self.pos_weight_fair = (len(self.data_fair.labels) - pos_fair) / pos_fair
		pos_biased = np.sum(self.data_biased.labels)
		self.pos_weight_biased = (len(self.data_biased.labels) - pos_biased) / pos_biased

		assert (val_fair is None) == (val_biased is None), "Either both validation datasets should be provided or none."
		assert (test_fair is None) == (test_fair is None), "Either both test datasets should be provided or none."
		assert (val_fair is None) == (test_fair is None), "Either both validation and test datasets should be provided or none."
		self.test_cv = test_fair is None
		self.val_fair = val_fair
		self.val_biased = val_biased
		self.test_fair = test_fair
		self.test_biased = test_biased

	def make_datasets(self, data_fair, data_biased, images):
		if images:
			data_fair = ImageSensitiveDataset(data_fair.instance_names,
											  data_fair.features,
											  data_fair.labels,
										   	  device=self.cfg.device)
			data_biased = ImageSensitiveDataset(data_biased.instance_names,
											 	data_biased.features,
											   	data_biased.labels,
											   	device=self.cfg.device)
			return data_fair, data_biased
		X_fair, y_fair = data_fair.features, data_fair.labels.ravel()
		data_fair = TabularDataset(torch.tensor(X_fair, dtype=torch.float32),
								   torch.tensor(y_fair, dtype=torch.int32),
								   self.cfg.sensitive_index,
								   feature_names=self.data_fair.feature_names)
		X_biased, y_biased = data_biased.features, data_biased.labels.ravel()
		data_biased = TabularDataset(torch.tensor(X_biased, dtype=torch.float32),
									 torch.tensor(y_biased, dtype=torch.int32),
									 self.cfg.sensitive_index,
									 feature_names=data_biased.feature_names)
		return data_fair, data_biased

	def test_classifier_given_test(self, classifier: Classifier2, train_biased, massaging=False):
		train_data_fair, train_data_biased = self.make_datasets(self.data_fair, self.data_biased, images=self.images)
		val_data_fair, val_data_biased = self.make_datasets(self.val_fair, self.val_biased, images=self.images)
		test_data_fair, test_data_biased = self.make_datasets(self.test_fair, self.test_biased, images=self.images)
		if train_biased:
			train_data = train_data_biased
			val_data = val_data_biased
		else:
			train_data = train_data_fair
			val_data = val_data_fair
		if massaging:
			masseuse = Massaging(sensitive_group=1)
			masseuse.fit(train_data)
			train_data = masseuse.transform(train_data)
			if val_data is not None:
				masseuse = Massaging(sensitive_group=1)
				masseuse.fit(val_data)
				val_data = masseuse.transform(val_data)
		classifier.fit(train_data, val_data)
		classifier.evaluate_biased(False)
		classifier.test(test_data_fair, store_idx=0)
		classifier.evaluate_biased(True)
		classifier.test(test_data_biased, store_idx=1)
		fair_results, biased_results = classifier.evaluate(0), classifier.evaluate(1)
		fair_df = pd.DataFrame(fair_results | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name])
		biased_df = pd.DataFrame(biased_results | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name])

		return fair_df, biased_df

	def test_postprocessing(self, classifier, train_data, test_data_fair, test_data_biased):
		DP_classifier = PostProcessingWrapper(classifier, 'demographic_parity')
		DP_classifier.enable_postprocessing(train_data)
		DP_classifier.evaluate_biased(False)
		DP_classifier.test(test_data_fair, store_idx=2)
		DP_classifier.evaluate_biased(True)
		DP_classifier.test(test_data_biased, store_idx=3)

		EO_classifier = PostProcessingWrapper(classifier, 'equalized_odds')
		EO_classifier.enable_postprocessing(train_data)
		EO_classifier.evaluate_biased(False)
		EO_classifier.test(test_data_fair, store_idx=4)
		EO_classifier.evaluate_biased(True)
		EO_classifier.test(test_data_biased, store_idx=5)

	def evaluate_postprocessing(self, classifier):
		fair_results_DP, biased_results_DP = classifier.evaluate(2), classifier.evaluate(3)
		fair_results_DP['Intervention'] += "-DP"
		biased_results_DP['Intervention'] += "-DP"
		fair_df_DP = pd.DataFrame(fair_results_DP | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name+"-DP"])
		biased_df_DP = pd.DataFrame(biased_results_DP | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name+"-DP"])

		fair_results_EO, biased_results_EO = classifier.evaluate(4), classifier.evaluate(5)
		fair_results_EO['Intervention'] += "-EO"
		biased_results_EO['Intervention'] += "-EO"
		fair_df_EO = pd.DataFrame(fair_results_EO | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name+"-EO"])
		biased_df_EO = pd.DataFrame(biased_results_EO | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name+"-EO"])
		return pd.concat([fair_df_DP, fair_df_EO]), pd.concat([biased_df_DP, biased_df_EO])

	def test_classifier(self, classifier: Classifier2, train_biased, massaging=False, postprocessing=False):
		if not self.test_cv:
			# if postprocessing:
			# 	raise NotImplementedError
			return self.test_classifier_given_test(classifier, train_biased, massaging)

		splitter = TorchStratifiedKFold(k=5, shuffle=True, random_state=self.cfg.seed)
		data_fair, data_biased = self.make_datasets(self.data_fair, self.data_biased, images=self.images)
		for train_sets, test_sets in splitter.split2([data_fair, data_biased]): # stratify according to fair labels
			test_data_fair, test_data_biased = test_sets
			if self.data_drift is not None:
				X_drift, y_drift = (self.data_drift.features, self.data_drift.labels.ravel())
				data_drift = TabularDataset(torch.tensor(X_drift, dtype=torch.float32),
											 torch.tensor(y_drift, dtype=torch.int32),
											 self.cfg.sensitive_index,
											 feature_names=self.data_drift.feature_names)
				test_data_fair = Subset(data_drift, test_data_fair.indices)
			if train_biased:
				train_data = train_sets[1]
			else:
				train_data = train_sets[0]
			if massaging:
				masseuse = Massaging(sensitive_group=1)
				masseuse.fit(train_data)
				train_data = masseuse.transform(train_data)
			classifier.fit(train_data)
			classifier.evaluate_biased(False)
			classifier.test(test_data_fair, store_idx=0)
			classifier.evaluate_biased(True)
			classifier.test(test_data_biased, store_idx=1)
			if postprocessing:
				self.test_postprocessing(classifier, train_data, test_data_fair, test_data_biased)
		fair_results, biased_results = classifier.evaluate(0), classifier.evaluate(1)
		fair_df = pd.DataFrame(fair_results | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name])
		biased_df = pd.DataFrame(biased_results | {"RunId" : f"{self.cfg.seed}-{self.cfg.rates}"}, index=[classifier.name])
		if postprocessing:
			fair_df_post, biased_df_post = self.evaluate_postprocessing(classifier)
			fair_df = pd.concat([fair_df, fair_df_post])
			biased_df = pd.concat([biased_df, biased_df_post])

		return fair_df, biased_df

	def test_upper_baseline(self):
		nn = NNClassifier(name="UpperBaseline",
						  cfg=self.cfg,
						  images=self.images,
						  input_features=self.nb_features,
						  drop_sensitive=False,
						  pos_weight=self.fair_pos_weight)
		fair_results, biased_results =  self.test_classifier(nn, train_biased=False)
		return fair_results, biased_results

	def test_lower_baseline(self):
		nn = NNClassifier(name="LowerBaseline",
						  cfg=self.cfg,
						  images=self.images,
						  input_features=self.nb_features,
						  drop_sensitive=False,
						  pos_weight=self.biased_pos_weight)
		fair_results, biased_results = self.test_classifier(nn, train_biased=True, postprocessing=True)
		return fair_results, biased_results

	def test_massaging(self):
		nn = NNClassifier(name="Massaging",
						  cfg=self.cfg,
						  images=self.images,
						  input_features=self.nb_features,
						  drop_sensitive=False,
						  pos_weight=self.biased_pos_weight)
		fair_results, biased_results = self.test_classifier(nn, train_biased=True, massaging=True)
		return fair_results, biased_results

	def test_unawareness(self):
		nn = NNClassifier(name="Unawareness",
						  cfg=self.cfg,
						  images=self.images,
						  input_features=self.nb_features,
						  drop_sensitive=True,
						  pos_weight=self.biased_pos_weight)
		fair_results, biased_results = self.test_classifier(nn, train_biased=True)
		return fair_results, biased_results

	def test_problog(self, fair_params=None):
		problog_classifier = ProbLogClassifier2(self.cfg, self.images, self.nb_features, fair_params=fair_params,
												pos_weight=self.biased_pos_weight)
		fair_results, biased_results = self.test_classifier(problog_classifier, train_biased=True)
		self.problog_biased_results.append(biased_results)
		self.problog_fair_results.append(fair_results)
		return fair_results, biased_results

	def test_baselines(self):
		self.baseline_biased_results = []
		self.baseline_fair_results = []

		if self.images:
			test_functions = [self.test_upper_baseline, self.test_lower_baseline]
		else:
			test_functions = [self.test_lower_baseline, self.test_upper_baseline, self.test_unawareness, self.test_massaging]
		for test_function in test_functions:
			fair_results, biased_results = test_function()
			self.baseline_biased_results.append(biased_results)
			self.baseline_fair_results.append(fair_results)
		biased_df = pd.concat(self.baseline_biased_results)
		fair_df = pd.concat(self.baseline_fair_results)
		return fair_df, biased_df

	def clear_problog_results(self):
		self.problog_biased_results = []
		self.problog_fair_results = []

	def set_cfg(self, cfg: ConfigDict):
		self.cfg = cfg

	def get_results(self, problog_only = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
		if problog_only:
			return pd.concat(self.problog_fair_results).reset_index().drop(columns=['index']),pd.concat(self.problog_biased_results).reset_index().drop(columns=['index']),
		fair_df = pd.concat(self.baseline_fair_results + self.problog_fair_results).reset_index().drop(columns=['index'])
		biased_df = pd.concat(self.baseline_biased_results + self.problog_biased_results).reset_index().drop(
			columns=['index'])
		return fair_df, biased_df
