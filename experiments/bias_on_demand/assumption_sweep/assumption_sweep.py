import os
from datetime import datetime
from pathlib import Path

from src.parameters import get_measurement_params, get_label_params
from experiments.bias_on_demand.dataset import make_standard, generate_dataset
from src import set_seed, load_config, GridSearch, ARTIFACTS_PATH
from src.testing import Tester
import pandas as pd

def set_label_rates(cfg):
	rates = get_label_params(cfg.rates[0], cfg.rates[1])
	names = [['tpr_not_a', 1], ['tpr_a', 1], ['fpr_not_a', 1], ['fpr_a', 1]]

	return rates, names, cfg.rates[1], cfg.p_l, '$p^{label}_Y$'

def set_measurement_rates(cfg):
	p_attrs = [cfg.p_r] + [cfg.p_q[i] + cfg.alpha_qr[i] * cfg.p_r for i in range(len(cfg.alpha_qr))]
	rates = sum([get_measurement_params(p_attr, cfg.rates[0], cfg.rates[1]) for p_attr in p_attrs], [])
	names = sum([[['ppv_not_a',i+1], ['ppv_a',i+1], ['npv_not_a',i+1], ['npv_a',i+1]] for i in range(len(p_attrs))], [])

	return rates, names, cfg.rates[1], cfg.p_m_r, '$p^{measure}_X$'

def set_historical_rates(cfg):
	if cfg.as_label:
		return set_label_rates(cfg)
	else:
		return set_measurement_rates(cfg)

def set_rates(cfg):
	"""
	Returns:
		- rates: list of rates for the model
		- names: list of names for the rates
		- p_bias: the bias parameter used in the rates
		- real_p_bias: the actual bias parameter used in the rates
		- x_label: label for the x-axis in the plot
	"""
	if 'historical' in cfg.name:
		return set_historical_rates(cfg)
	elif 'label' in cfg.name:
		return set_label_rates(cfg)
	elif 'measurement' in cfg.name:
		return set_measurement_rates(cfg)
	else:
		raise ValueError("Unknown rates configuration in name: " + cfg.name)

def run(cfg_path):
	# load cfg
	cfg = load_config(cfg_path)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_nb)
	dataset_cfg = load_config(cfg.dataset_cfg_path)
	cfg.update(dataset_cfg)

	# init
	biased_lst = []
	fair_lst = []

	# init log results
	path = Path(ARTIFACTS_PATH) / Path(f"{cfg.project}/{cfg.name}/" + datetime.now().strftime("%d%m%Y-%H%M%S"))
	path.mkdir(parents=True, exist_ok=True)

	# loop over seeds
	for cfg1 in GridSearch(cfg, ['seed']):
		set_seed(cfg1.seed)

		# load data
		X = generate_dataset(dataset_cfg, **dataset_cfg)
		tester = Tester(make_standard(X, fair=True), make_standard(X, fair=False), cfg=cfg)
		tester.set_cfg(cfg1)
		# tester.test_baselines()

		# loop over rates
		for cfg2 in GridSearch(cfg1, ['rates']):
			cfg2['rates'], cfg2['names'], p_bias, real_p_bias, x_label = set_rates(cfg2)

			cfg2.add({'p_bias': p_bias})
			tester.set_cfg(cfg2)
			tester.clear_problog_results()
			tester.test_problog()

			fair_df, biased_df = tester.get_results()
			info = {'Seed': [cfg2.seed]*len(biased_df), 'Param': [p_bias]*len(biased_df)}
			biased_df = biased_df.assign(**info)
			fair_df = fair_df.assign(**info)
			biased_lst.append(biased_df)
			fair_lst.append(fair_df)

			pd.concat(fair_lst).to_csv(path / f"all_fair.csv", index=False)
			pd.concat(biased_lst).to_csv(path / f"all_biased.csv", index=False)
			cfg2.save(path / "config.yaml")
