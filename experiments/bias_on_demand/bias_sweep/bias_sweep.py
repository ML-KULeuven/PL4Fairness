from datetime import datetime
from pathlib import Path

import pandas as pd
import os

from src.parameters import get_label_params, estimate_measurement_params, estimate_historical_params
from experiments.bias_on_demand.dataset import make_standard, generate_dataset
from src import load_config, GridSearch, set_seed, ARTIFACTS_PATH
from src.testing import Tester

# bias type is the prefix of cfg.name; variants may add a suffix (e.g. "_indep_a")
def set_rates(train_X, cfg):
	if cfg.name.startswith("label"):
		cfg['rates'] = get_label_params(cfg.noise_py, cfg.p_l)
	elif cfg.name.startswith("measurement"):
		cfg['rates'] = estimate_measurement_params(train_X, cfg.n_q)
	elif cfg.name.startswith("historical"):
		cfg['rates'] = estimate_historical_params(train_X, cfg.n_q, cfg.as_label)
	else:
		raise ValueError(f"Unknown cfg.name: {cfg.name}")
	return cfg

def get_grid_param(cfg):
	if cfg.name.startswith("label"):
		return 'p_l', '$p^{label}_Y$'
	elif cfg.name.startswith("measurement"):
		return 'p_m_r', '$p^{measure}$'
	elif cfg.name.startswith("historical"):
		return 'p_h_r', '$p^{hist}$'
	else:
		raise ValueError(f"Unknown cfg.name: {cfg.name}")

def run(cfg_path):
	# load cfg
	cfg = load_config(cfg_path)
	dataset_cfg = load_config(cfg.dataset_cfg_path)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_nb)

	# init
	biased_lst = []
	fair_lst = []

	# init log results
	path = ARTIFACTS_PATH / Path(f"{cfg.project}/{cfg.name}/" + datetime.now().strftime("%d%m%Y-%H%M%S"))
	path.mkdir(parents=True, exist_ok=True)

	# loop over seeds
	grid_param, x_label = get_grid_param(cfg)
	for cfg2 in GridSearch(cfg, ['seed']):
		for dataset_cfg2 in GridSearch(dataset_cfg, [grid_param]):
			dataset_cfg2['p_m_q'] = [dataset_cfg2.p_m_r] *  dataset_cfg2.n_q # p_m_q = p_m_r for all Qs
			dataset_cfg2['p_h_q'] = [dataset_cfg2.p_h_r] * dataset_cfg2.n_q  # p_h_q = p_h_r for all Qs
			cfg2.update(dataset_cfg2)
			set_seed(cfg2.seed)

			# load data
			X = generate_dataset(dataset_cfg2, **dataset_cfg2)
			cfg2 = set_rates(X, cfg2)
			tester = Tester(make_standard(X, fair=True), make_standard(X, fair=False), cfg=cfg2)
			tester.set_cfg(cfg2)

			# test baselines
			tester.test_problog()
			tester.test_baselines()

			# save results
			fair_df, biased_df = tester.get_results()
			info = {'Seed': [cfg2.seed]*len(biased_df), 'Param': [cfg2[grid_param]]*len(biased_df)}
			biased_df = biased_df.assign(**info)
			fair_df = fair_df.assign(**info)
			biased_lst.append(biased_df)
			fair_lst.append(fair_df)
			pd.concat(fair_lst).to_csv(path / f"all_fair.csv", index=False)
			pd.concat(biased_lst).to_csv(path / f"all_biased.csv", index=False)
			cfg2.save(path / "config.yaml")
