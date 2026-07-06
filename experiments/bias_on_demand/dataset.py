from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.parameters import estimate_precisions, estimate_npvs
from src import load_config, ConfigDict, GridSearch
from src.metrics import get_all_metrics
from src.utils import make_standard, split_dataset


class ListDataset(Dataset):
	def __init__(self, list_length, *args):
		self.n = list_length

		self.value_check(args)
		self.data = args

	def value_check(self, args):
		for arg in args:
			if len(arg) != len(args[0]):
				raise ValueError(f"All inputs must have the same length. {len(arg)} != {len(args[0])}")


	def __getitem__(self, index):
		data_list = len(self.data)*[[]]
		offset = self.n * index

		for i in range(self.n):
			for j in range(len(self.data)):
				data_list[j] = self.data[j][offset+i]

		return tuple(data_list)

	def __len__(self):
		return len(self.data[0]) // self.n

def save_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, path: Path):
	# make parent directory
	Path(path).mkdir(parents=True, exist_ok=True)

	# save the generated dataset as csv
	train_df.to_csv(path.joinpath('X_train.csv'))
	val_df.to_csv(path.joinpath('X_val.csv'))
	test_df.to_csv(path.joinpath('X_test.csv'))

def generate_a(p: float, dim: int):
	"""
	Parameters:
		p: binomial probability of a
		dim: number of rows
	"""
	return np.random.binomial(1, p, dim)

def generate_r(p: float, p_h, A: np.array):
	"""
	Parameters:
		p: binomial probability of r
		p_h: coefficient of historical bias
		A: sensitive features
	"""
	r_fair = np.random.binomial(1, p, len(A))
	Flip = np.random.binomial(1, [p_h*a for a in A], len(A))

	return np.array([r*(1-f) for r, f in zip(r_fair, Flip)]), r_fair

def generate_q(p: float, alpha: float, R: np.array, p_h: float, A: np.array):
	"""
	Parameters:
		p: binomial probability of q
		alpha: additional probability of q given X = 1
		R: resource values
		p_h: coefficient of bias
		A: sensitive features
	"""
	assert len(A) == len(R)

	ps = [p + alpha * r for r in R]
	q_fair = np.random.binomial(1, ps, len(A))
	Flip = np.random.binomial(1, [p_h*a for a in A], len(A))

	return np.array([q*(1-f) for q, f in zip(q_fair, Flip)]), q_fair

def generate_proxy(X: np.array, p_noise: float, p_bias: float, A: np.array):
	"""
	Parameters:
		X: original values
		p_noise: probability of noise
		p_bias: probability of bias
		A: sensitive features
	"""
	assert len(A) == len(X)
	Noise = np.random.binomial(1, p_noise, len(X))
	Flip = np.random.binomial(1, [p_bias*a for a in A], len(A))

	biased_X = [x*(1-f) for x, f in zip(X, Flip)]
	noisy_biased_X = [(1-x)*n + x*(1-n) for x, n in zip(biased_X, Noise)]

	return np.array(noisy_biased_X)

def xor(a, b):
	return a*(1-b) + (1-a)*b

def _sanity_check(df, p_a, cfg: ConfigDict):
	print('Mixing biases breaks independence assumptions, so sanity check might fail.')
	print("Expected		| Found")

	mean_A = p_a
	print(f"A: 	{mean_A:.2f} ({p_a * (1-p_a):.2f})	| {df['A'].mean():.2f} ({df['A'].var():.2f})")

	mean_R_fair = cfg.p_r
	mean_R = mean_R_fair * (1 - cfg.p_h_r * p_a)
	mean_PR = xor(mean_R * (1 - cfg.p_m_r * p_a), cfg.noise_pr)

	print(f"R: 	{mean_R:.2f} 			| {df[f'R'].mean():.2f} ({df[f'R'].var():.2f})")
	print(f"PR: {mean_PR:.2f} 		| {df[f'PR'].mean():.2f} ({df[f'PR'].var():.2f})")

	mean_Q_fair = []
	mean_Q = []
	mean_PQ = []
	for i in range(cfg.n_q):
		mean_Q_fair.append(cfg.p_q[i] + cfg.alpha_qr[i] * mean_R_fair)
		mean_Q.append(mean_Q_fair[i] * (1 - cfg.p_h_q[i] * p_a))
		mean_PQ.append(xor(mean_Q[i] * (1 - cfg.p_m_q[i]*p_a), cfg.noise_pq[i]) )

		print(f"Q{i}: {mean_Q[i]:.2f} 		| {df[f'Q{i}'].mean():.2f} ({df[f'Q{i}'].var():.2f})")
		print(f"PQ{i}: {mean_PQ[i]:.2f} 		| {df[f'PQ{i}'].mean():.2f} ({df[f'Q{i}'].var():.2f})")

	mean_S_fair = cfg.alpha_a * mean_A + cfg.alpha_r * mean_R_fair + sum([mean_Q_fair[i] * cfg.alpha_q[i] for i in range(cfg.n_q)])
	mean_S = cfg.alpha_a * mean_A + cfg.alpha_r * mean_R + sum([mean_Q[i] * cfg.alpha_q[i] for i in range(cfg.n_q)])

	print(f"S_fair: {mean_S_fair:.2f}	| {df['S_fair'].mean():.2f} ({df['S_fair'].var():.2f})")
	print(f"S: 	{mean_S:.2f}			| {df['S'].mean():.2f} ({df['S'].var():.2f})")

	mean_Y_fair = df['Y_fair'].mean()
	mean_Y = xor(mean_Y_fair, cfg.noise_py) * (1 - cfg.p_l * p_a)

	print(f"Y:	{mean_Y:.2f}			| {df['Y'].mean():.2f} ({df['Y'].var():.2f})")
	print("==="*10)


def sanity_check(df: pd.DataFrame, cfg: ConfigDict):
	_sanity_check(df, cfg.p_a, cfg)
	_sanity_check(df[df['A'] == 0], 0, cfg)
	_sanity_check(df[df['A'] == 1], 1, cfg)

def generate_dataset(cfg,
					 n_q: int,
					 p_a: float,
					 p_r: float,
					 p_q: List[float],
					 std_s: float,
					 threshold_s: float,
					 alpha_a: float,
					 alpha_qr: List[float],
					 alpha_r: float,
					 alpha_q: List[float],
					 p_h_r: float,
					 p_h_q: List[float],
					 p_m_r: float,
					 p_m_q: List[float],
					 p_l: float,
					 noise_pr: float,
					 noise_pq: list[float],
					 noise_py : float,
					 dim: int = 1000,
					 dataset_path: str = 'datasets/dataset',
					 seed: int = None):
	"""Generate a dataset with the following characteristics:

	Parameters:
		n_q (int): number of Q-features => length of all lists
		p_a (float): probability of A = 1
		p_r (float): probability of R = 1
		p_q (List[float]): list of probabilities of Q_i = 1
		std_s (float): standard deviation of the noise on S
		threshold_s (float): threshold for S to determine Y

		alpha_qr (List[float]): list of additional probabilities of Q_i = 1 given R = 1
		alpha_a (float): weights of A = 1 to determine S (non bias influence of sensitive feature)
		alpha_r (float): weight of R = 1 to determine S
		alpha_q (List[float]): weight of Q_i = 1 to determine S

		p_h_r (float): probability of historical bias on R
		p_h_q (List[float]): probabilities of historical bias on Q_i

		p_m_r (float): probability of measurement bias on R
		p_m_q (List[float]): probability of measurement bias on Q_i

		p_l (float): probability of label bias on Y

		noise_pr (float): probability of random flip on pr 			=> 0.5 makes it random
		noise_pq (List[float]): probability of random flip on pq_i 	=> 0.5 makes it random
		noise_py (float): probability of random flip on py 			=> 0.5 makes it random

		dim (int): number of rows in the dataset
		splits (list): list of proportions for train, val and test sets (sums to 1)
		dataset_path (str): path of directory to save the dataset
		seed (int): seed for random number generations
	Saves:
		X_train: training set with R, Q, A, Y and Y_fair to {path}
		X_val: validation set with R, Q, A, Y and Y_fair to {path}
		X_test: test set with R, Q, A, Y and Y_fair to {path}
	"""

	# value checks
	assert len(p_q) == n_q, "p_q must have the same length as n_q"
	assert len(alpha_qr) == n_q, "alpha_rq must have the same length as n_q"
	assert len(noise_pq) == n_q, "alpha_pq must have the same length as n_q"
	assert len(alpha_q) == n_q, "alpha_q must have the same length as n_q"
	assert len(p_h_q) == n_q, "p_h_q must have the same length as n_q"
	assert len(p_m_q) == n_q, "p_m_q must have the same length as n_q"

	assert all([p_q[i] + alpha_qr[i] <= 1 for i in range(n_q)]), "p_q + alpha_qr must be <= 1"

	# set seed for reproducibility
	if seed is not None:
		np.random.seed(seed)

	# generate dataset
	A = generate_a(p_a, dim)
	R, R_fair = generate_r(p_r, p_h_r, A)
	Q, Q_fair = tuple(zip(*[generate_q(p_q[i], alpha_qr[i], R, p_h_q[i], A) for i in range(n_q)]))

	noise_S = np.random.normal(0, std_s, dim)
	S = alpha_a * (1-A) + alpha_r * R + np.sum([alpha_q[i] * Q[i] for i in range(n_q)], axis=0) + noise_S
	S_fair = alpha_a * (1-A) + alpha_r * R_fair + np.sum([alpha_q[i] * Q_fair[i] for i in range(n_q)], axis=0) + noise_S

	# generate proxies
	PR = generate_proxy(R, noise_pr, p_m_r, A)
	PQ = np.array([generate_proxy(Q[i], noise_pq[i], p_m_q[i], A) for i in range(n_q)])

	# label
	Y_fair = np.where(S_fair >= threshold_s, 1, 0) # without historical bias
	Y = np.where(S >= threshold_s, 1, 0)		   # with historical bias
	PY = generate_proxy(Y, noise_py, p_l, A)  	   # with historical and label bias

	# create dataframe
	X = pd.DataFrame(
		{'A': A, 'R': R, 'PR': PR, 'R_no_hist': R_fair} |
		{f'Q{i}': Q[i] for i in range(n_q)} |
		{f'PQ{i}': PQ[i, :] for i in range(n_q)} |
		{f'Q{i}_no_hist': Q_fair[i] for i in range(n_q)} |
		{'S': S, 'S_fair': S_fair, 'Y_fair': Y_fair, 'Y': PY})
	# sanity check
	sanity_check(X, cfg)

	X = X.drop(['S', 'S_fair'], axis=1)

	# split into train, val and test sets
	path = Path(dataset_path)
	Path(path).mkdir(parents=True, exist_ok=True)
	cfg.save(path / 'config.yaml')
	with open(path / 'stats.txt', 'w') as f:
		f.write(get_metrics(X, 'X'))
		f.write("===== R - STATS =====")
		f.write(get_metrics(X, 'X', 'R'))
		for i in range(cfg.n_q):
			f.write(f"\n===== Q{i} - STATS =====")
			f.write(get_metrics(X, f'X', f'Q{i}'))
	return X

def get_metrics(df, name, label_name="Y"):
	biased_data = make_standard(df, fair=False, label_name=label_name)
	fair_data = make_standard(df, fair=True, label_name=label_name)
	metrics = get_all_metrics(biased_data, fair_data)

	str = f"{name} METRICS\n"+"==="*10 + "\n"
	str += "Positive rate (tot/priv/non-priv): {:.2f} / {:.2f} / {:.2f}".format(metrics["Positive Rate"], metrics["Privileged Positive Rate"], metrics["Unprivileged Positive Rate"]) + "\n"
	str += "Statistical Disparity: {:.3f}".format(metrics["Statistical Disparity"]) + "\n"
	str += "TPR difference: {:.2f}".format(metrics["TPR_diff"]) + "\n"
	str += "FPR difference: {:.2f}".format(metrics["FPR_diff"]) + "\n"
	if label_name == "Y":
		str += "TPRs (priv/non-priv): {:.2f} / {:.2f}".format(metrics['Privileged TPR'],
															  metrics['Unprivileged TPR']) + "\n"
		str += "FPRs (priv/non-priv): {:.2f} / {:.2f}\n\n".format(metrics['Privileged FPR'],
															  metrics['Unprivileged FPR']) + "\n"
	else:
		str += "PPV (tot/priv/non-priv): {:.2f} / {:.2f} / {:.2f} ".format(*estimate_precisions(df, label_name, "P" + label_name, "A")) + "\n"
		str += "NPV (tot/priv/non-priv): {:.2f} / {:.2f} / {:.2f} \n\n".format(
			*estimate_npvs(df, label_name, "P" + label_name, "A")) + "\n"

	return str

def generate_all(dataset_path, save_path, param):
	dataset_cfg = load_config(dataset_path)
	for seed in range(0, 10):
		dataset_cfg['seed'] = seed
		for dataset_cfg2 in GridSearch(dataset_cfg, [param]):
			dataset_cfg2['dataset_path'] = save_path + f"/seed{seed}/bias{dataset_cfg2[param]}"
			generate_dataset(dataset_cfg2, **dataset_cfg2)
