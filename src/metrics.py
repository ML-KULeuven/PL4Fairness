import torch
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def get_groups(dataset: StandardDataset):
	attr_names = dataset.protected_attribute_names
	priv_values = dataset.privileged_protected_attributes
	unpriv_values = dataset.unprivileged_protected_attributes

	priv_groups = sum([[{a: c} for c in c_list] for a, c_list in zip(attr_names, priv_values)], [])
	unpriv_groups = sum([[{a: c} for c in c_list] for a, c_list in zip(attr_names, unpriv_values)], [])

	return priv_groups, unpriv_groups

def probabilistic_statistical_disparity(y_pred: torch.Tensor, sensitive_attr: torch.Tensor, privileged_value=0):
	mean_prob_priv = torch.mean(y_pred[sensitive_attr == privileged_value])
	mean_prob_unpriv = torch.mean(y_pred[sensitive_attr != privileged_value])
	return (mean_prob_unpriv - mean_prob_priv).item()

def probabilistic_equalized_opportunity_disparity(y_pred: torch.Tensor, y_true: torch.Tensor, sensitive_attr: torch.Tensor, privileged_value=0, favorable_value=1):
	mean_tpr_priv = torch.mean(y_pred[torch.logical_and(sensitive_attr == privileged_value,y_true == favorable_value)])
	mean_tpr_unpriv = torch.mean(y_pred[torch.logical_and(sensitive_attr != privileged_value,y_true == favorable_value)])
	return (mean_tpr_unpriv - mean_tpr_priv).item()

def probabilistic_equalized_odds_disparity(y_pred: torch.Tensor, y_true: torch.Tensor, sensitive_attr: torch.Tensor, privileged_value=0, favorable_value=1):
	correct_priv = torch.concatenate((y_pred[torch.logical_and(sensitive_attr == privileged_value,y_true == favorable_value)],
									  1-y_pred[torch.logical_and(sensitive_attr == privileged_value, y_true == 1-favorable_value)]))
	mean_correct_priv = torch.mean(correct_priv)
	correct_unpriv = torch.concatenate((y_pred[torch.logical_and(sensitive_attr != privileged_value,y_true == favorable_value)],
										1-y_pred[torch.logical_and(sensitive_attr != privileged_value,y_true == 1 - favorable_value)]))
	mean_correct_unpriv = torch.mean(correct_unpriv)
	return (mean_correct_unpriv - mean_correct_priv).item()

def get_all_metrics(biased_dataset: StandardDataset, fair_dataset: StandardDataset):
	# make features the same to cheat check in StandardDataset
	biased_dataset.features = fair_dataset.features

	privileged_groups, unprivileged_groups = get_groups(fair_dataset)
	dataset = ClassificationMetric(fair_dataset, biased_dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
	measures = {"Privileged "+k: v for k, v in dataset.performance_measures(privileged=True).items()}
	measures.update({"Unprivileged "+k:v for k, v in dataset.performance_measures(privileged=False).items()})
	measures.update({"TPR_diff": dataset.true_positive_rate_difference(),
					"FPR_diff": dataset.false_positive_rate_difference()})
	dataset = BinaryLabelDatasetMetric(biased_dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
	measures.update({"Positive Rate": dataset.base_rate(),
			"Privileged Positive Rate": dataset.base_rate(True),
			"Unprivileged Positive Rate": dataset.base_rate(False),
			"Statistical Disparity": dataset.statistical_parity_difference()})
	return measures