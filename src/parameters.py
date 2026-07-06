import pandas as pd
import sklearn

from src.metrics import get_all_metrics
from src.utils import make_standard


def get_label_params(p_noise, p_bias):
	p_flip = p_bias * (1-p_noise) + p_noise * (1-p_bias)
	# tpr_not_a, tpr_a, fpr_not_a, fpr_a
	return [(1-p_noise), 1-p_flip, p_noise, p_noise]

def get_ppv(p_attr, p_noise, p_bias):
	p_flip_attr = p_bias * (1 - p_noise) + p_noise * (1 - p_bias)
	p_flip_not_attr = p_noise

	divider =  (p_attr * (1-p_flip_attr) + (1-p_attr) * p_flip_not_attr)
	if divider == 0:
		return 0

	return p_attr * (1-p_flip_attr) / divider

def get_npv(p_attr, p_noise, p_bias):
	p_flip_attr = p_bias * (1 - p_noise) + p_noise * (1 - p_bias)
	p_flip_not_attr = p_noise

	divider = (1-p_attr) * (1-p_flip_not_attr) + p_attr * p_flip_attr
	if divider == 0:
		return 0

	return (1-p_attr) * (1-p_flip_not_attr) / divider

def get_measurement_params(p_attr, p_noise, p_bias):
	# ppv_not_a, ppv_a, npv_not_a, npv_a
	return [get_ppv(p_attr, p_noise, 0), get_ppv(p_attr, p_noise, p_bias),
			get_npv(p_attr, p_noise, 0), get_npv(p_attr, p_noise, p_bias)]

def estimate_label_params(train_X):
	biased_data = make_standard(train_X, fair=False, label_name='Y')
	fair_data = make_standard(train_X, fair=True, label_name='Y')
	metrics = get_all_metrics(biased_data, fair_data)
	return [metrics['Privileged TPR'], metrics['Unprivileged TPR'], metrics['Privileged FPR'], metrics['Unprivileged FPR']]

def estimate_precisions(df: pd.DataFrame, fair_col, biased_col, sensitive_col):
	tot_pr = sklearn.metrics.precision_score(df[fair_col], df[biased_col])
	priv_pr = sklearn.metrics.precision_score(df[df[sensitive_col] == 0][fair_col], df[df[sensitive_col] == 0][biased_col])
	non_priv_pr = sklearn.metrics.precision_score(df[df[sensitive_col] == 1][fair_col], df[df[sensitive_col] == 1][biased_col])

	return tot_pr, priv_pr, non_priv_pr

# negative predictive value: TN / (FN + TN)
def estimate_npvs(df: pd.DataFrame, fair_col, biased_col, sensitive_col):
	tot_pr = sklearn.metrics.precision_score(df[fair_col], df[biased_col], pos_label=0)
	priv_pr = sklearn.metrics.precision_score(df[df[sensitive_col] == 0][fair_col], df[df[sensitive_col] == 0][biased_col], pos_label=0)
	non_priv_pr = sklearn.metrics.precision_score(df[df[sensitive_col] == 1][fair_col], df[df[sensitive_col] == 1][biased_col], pos_label=0)

	return tot_pr, priv_pr, non_priv_pr

def estimate_measurement_params(train_X, n_q):
	lst = []
	_, ppv_not_a, ppv_a = estimate_precisions(train_X, fair_col='R', biased_col='PR', sensitive_col='A')
	_, npv_not_a, npv_a = estimate_npvs(train_X, fair_col='R', biased_col='PR', sensitive_col='A')
	lst += [ppv_not_a, ppv_a, npv_not_a, npv_a]
	for i in range(0, n_q):
		_, ppv_not_a, ppv_a = estimate_precisions(train_X, fair_col=f'Q{i}', biased_col=f'PQ{i}', sensitive_col='A')
		_, npv_not_a, npv_a = estimate_npvs(train_X, fair_col=f'Q{i}', biased_col=f'PQ{i}', sensitive_col='A')
		lst += [ppv_not_a, ppv_a, npv_not_a, npv_a]
	return lst

def estimate_historical_params(train_X, n_q, as_label):
	if as_label:
		return estimate_label_params(train_X)
	else:
		lst = []
		_, ppv_not_a, ppv_a = estimate_precisions(train_X, fair_col='R_no_hist', biased_col='R', sensitive_col='A')
		_, npv_not_a, npv_a = estimate_npvs(train_X, fair_col='R_no_hist', biased_col='R', sensitive_col='A')
		lst += [ppv_not_a, ppv_a, npv_not_a, npv_a]
		for i in range(0, n_q):
			_, ppv_not_a, ppv_a = estimate_precisions(train_X, fair_col=f'Q{i}_no_hist', biased_col=f'Q{i}', sensitive_col='A')
			_, npv_not_a, npv_a = estimate_npvs(train_X, fair_col=f'Q{i}_no_hist', biased_col=f'Q{i}', sensitive_col='A')
			lst += [ppv_not_a, ppv_a, npv_not_a, npv_a]
		return lst




