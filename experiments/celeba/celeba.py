import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.celeba.dataset import load_celeba
from src import load_config, GridSearch, set_seed, ARTIFACTS_PATH
from src.testing import Tester

from sklearn.metrics import confusion_matrix

def get_tpr_fpr(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	tn = cm[0, 0]
	fp = cm[0, 1]
	fn = cm[1, 0]
	tp = cm[1, 1]
	return tp / (tp + fn), fp / (tn + fp)

def estimate_label_params2(fair_data, biased_data, feature_idx, priv_val=0):
	non_priv_val = (1 - priv_val) % 2
	priv_idxs = (fair_data.features[:, feature_idx] == priv_val)
	unpriv_idxs = (fair_data.features[:, feature_idx] == non_priv_val)
	priv_tpr, priv_fpr = get_tpr_fpr(fair_data.labels.flatten()[priv_idxs], biased_data.labels.flatten()[priv_idxs])
	unpriv_tpr, unpriv_fpr = get_tpr_fpr(fair_data.labels.flatten()[unpriv_idxs], biased_data.labels.flatten()[unpriv_idxs])
	return priv_tpr, unpriv_tpr, priv_fpr, unpriv_fpr

def run(cfg_path):
	# setup
	cfg = load_config(cfg_path)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_nb)
	fair_lst, biased_lst = [], []
	tester = Tester(*load_celeba(cfg.protected_attr, cfg.to_keep), cfg=cfg, images=True)

	# init log results
	path = ARTIFACTS_PATH / Path(f"{cfg.project}/{cfg.name}/" + datetime.now().strftime("%d%m%Y-%H%M%S"))
	path.mkdir(parents=True, exist_ok=True)

	for i in range(1 + len(cfg.to_keep)):
		cfg['rates'] += estimate_label_params2(tester.data_fair, tester.data_biased, i)

	# loop over seeds
	for cfg1 in GridSearch(cfg, ['seed']):
		set_seed(cfg1.seed)
		tester.set_cfg(cfg1)
		tester.test_problog()
		tester.test_baselines()
		fair_df, biased_df = tester.get_results()
		fair_lst.append(fair_df.assign(**{'Seed': [cfg1.seed] * len(fair_df)}))
		biased_lst.append(biased_df.assign(**{'Seed': [cfg1.seed] * len(fair_df)}))

		pd.concat(fair_lst).to_csv(path / f"all_fair.csv", index=False)
		pd.concat(biased_lst).to_csv(path / f"all_biased.csv", index=False)
		cfg1.save(path / "config.yaml")
