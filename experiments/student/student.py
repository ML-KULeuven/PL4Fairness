import os
from datetime import datetime
from pathlib import Path
import pandas as pd

from src import set_seed, load_config, GridSearch, ARTIFACTS_PATH
from src.testing import Tester

from experiments.student.dataset import load_data_with_biased_and_unbiased_grades, preprocess_data

def run(cfg_path):
	# load cfg
	cfg = load_config(cfg_path)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_nb)

	# init
	biased_lst = []
	fair_lst = []

	# init log results
	path = ARTIFACTS_PATH / Path(f"{cfg.project}/{cfg.name}/" + datetime.now().strftime("%d%m%Y-%H%M%S"))
	path.mkdir(parents=True, exist_ok=True)


	# load and preprocess data
	data, categorical_attributes = load_data_with_biased_and_unbiased_grades()
	tester = Tester(*preprocess_data(data, categorical_attributes), cfg=cfg)
	names = ['correct',
			 #'no_neg_flips_not_a',  # tpr_not_a = 1
			 #'no_neg_flips_a',	# tpr_a = 1
			 #'no_pos_flips_not_a',  # fpr_not_a = 0
			 #'no_pos_flips_a',  # fpr_a = 0
			 #'no_bias_not_a', # tpr_not_a = 1, fpr_not_a = 0
			 'no_bias_not_a', # tpr_not_a = 1, fpr_not_a = 0
			 'no_pos_flips',  # fpr_not_a = 0, fpr_a = 0
			 ]

	# loop over seeds
	for cfg1 in GridSearch(cfg, ['seed']):
		tester.set_cfg(cfg1)
		set_seed(cfg1.seed)
		tester.test_baselines()
		fair_df, biased_df = tester.get_results()
		info = {'Seed': [cfg1.seed] * len(biased_df), 'Name': fair_df['Intervention'].tolist()}
		biased_lst.append(biased_df.assign(**info))
		fair_lst.append(fair_df.assign(**info))
		# loop over rates
		for i,cfg2 in enumerate(GridSearch(cfg1, ['rates'])):
			tester.set_cfg(cfg2)
			tester.test_problog()
			fair_df, biased_df = tester.get_results(problog_only=True)
			info = {'Seed': [cfg2.seed]*len(biased_df), 'Name': [names[i]]*len(biased_df)}
			biased_lst.append(biased_df.assign(**info))
			fair_lst.append(fair_df.assign(**info))
			tester.clear_problog_results()

			pd.concat(fair_lst).to_csv(path / f"all_fair.csv", index=False)
			pd.concat(biased_lst).to_csv(path / f"all_biased.csv", index=False)
			cfg2.save(path / "config.yaml")
