from pathlib import Path
from typing import Tuple

import numpy as np
import torchvision
from aif360.datasets import StandardDataset
import pandas as pd

DATA_ROOT = Path("res/data/celeba")

def get_df(data: torchvision.datasets.VisionDataset, to_keep=['High_Cheekbones']) -> pd.DataFrame:
	"""
	Convert torchvision dataset to pandas DataFrame
	"""
	df = pd.DataFrame(data.attr, index=data.filename, columns=data.attr_names[:-1])
	df = df[['Mouth_Slightly_Open'] + to_keep]
	return df

def make_unbiased(df: pd.DataFrame, unbiased_labels) -> Tuple[pd.DataFrame,pd.DataFrame]:
	df = df[df.index.isin(unbiased_labels.index)] # drop images not in unbiased_labels
	unbiased_labels['Mouth_Slightly_Open'] = np.floor_divide(unbiased_labels['Mouth_Slightly_Open'].to_numpy() + 1, 2) # -1, 1 -> 0, 1
	df_ = df.copy()
	df_['Mouth_Slightly_Open'] = unbiased_labels['Mouth_Slightly_Open']
	return df, df_

def make_standard(df: pd.DataFrame, protected_attrs=["High_Cheekbones"], label_name='Mouth_Slightly_Open') -> StandardDataset:
	if not isinstance(protected_attrs,list):
		protected_attrs = [protected_attrs]
	return StandardDataset(
		df,
		label_name=label_name,
		favorable_classes=[1],
		protected_attribute_names=protected_attrs,
		privileged_classes=[[0]]*len(protected_attrs),
	)

def load_celeba(protected_attr='High_Cheekbones', to_keep=[]) -> Tuple[StandardDataset, StandardDataset, StandardDataset, StandardDataset, StandardDataset, StandardDataset]:
	"""
	Returns:
	- data_train: aif360.datasets.StandardDataset training original
	- data_train_fair: aif360.datasets.StandardDataset training unbiased
	- data_val: aif360.datasets.StandardDataset validation original
	- data_val_fair: aif360.datasets.StandardDataset validation unbiased
	- data_test: aif360.datasets.StandardDataset testing original
	- data_test_fair: aif360.datasets.StandardDataset testing unbiased
	"""
	# torchvision.datasets.VisionDataset
	data_train = torchvision.datasets.CelebA(DATA_ROOT, 'train', target_type='attr', download=True)
	data_val = torchvision.datasets.CelebA(DATA_ROOT, 'valid', target_type='attr', download=True)
	data_test = torchvision.datasets.CelebA(DATA_ROOT, 'test', target_type='attr', download=True)

	# pd.DataFrame
	unbiased_labels = pd.read_csv(DATA_ROOT / 'label_cleaned_mso.csv', index_col='image_id')
	unbiased_labels.index.name = 'image_id'

	df_train = get_df(data_train, [protected_attr] + to_keep)
	df_val = get_df(data_val, [protected_attr] + to_keep)
	df_test = get_df(data_test, [protected_attr] + to_keep)

	df_train, df_train_unbiased = make_unbiased(df_train, unbiased_labels)
	df_val, df_val_unbiased = make_unbiased(df_val, unbiased_labels)
	df_test, df_test_unbiased = make_unbiased(df_test, unbiased_labels)

	# cols = df_train.columns.to_list()
	# cols.remove('Mouth_Slightly_Open')
	# for c in cols:
	# 	metrics = get_all_metrics(make_standard(df_train, c, label_name='Mouth_Slightly_Open'), make_standard(df_train_unbiased, c, label_name='Mouth_Slightly_Open'))
	# 	print(f"{c} - SD {metrics['Statistical Disparity']}, tpr_not_a: {metrics['Privileged TPR']}, tpr_a: {metrics['Unprivileged TPR']}, fpr_not_a: {metrics['Privileged FPR']}, fpr_not_a: {metrics['Unprivileged FPR']}")

	return (make_standard(df_train_unbiased, [protected_attr]), make_standard(df_train, [protected_attr]),
			make_standard(df_val_unbiased, [protected_attr]), make_standard(df_val, [protected_attr]),
		 	make_standard(df_test_unbiased, [protected_attr]), make_standard(df_test, [protected_attr]))