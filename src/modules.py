import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score, binary_auroc, \
	binary_average_precision

from src.build_module import build_module
from src.metrics import get_all_metrics, probabilistic_statistical_disparity, probabilistic_equalized_odds_disparity, \
	probabilistic_equalized_opportunity_disparity
from src.training import FocalLoss, BalancedBCELoss, BrierScore
from src.utils import make_standard_from_torch, adapt_for_finetuning

class MyLightningModule(pl.LightningModule):
	def __init__(self, cfg, images=False, input_features=5, drop_sensitive=False, pos_weight=1):
		super().__init__()
		self.learning_rate = float(cfg.lr)
		self.sensitive_index = cfg.sensitive_index
		if cfg['loss'] == 'bce':
			self.loss = BalancedBCELoss(pos_weight=pos_weight)
		else:
			self.loss = FocalLoss(alpha=cfg['alpha'], gamma=cfg['gamma'])
		self.acc = binary_accuracy
		self.f1 = binary_f1_score
		self.auroc = binary_auroc
		self.ap = binary_average_precision
		self.brier = BrierScore()
		self.ece = BinaryCalibrationError()
		self.input_features = input_features
		self.images = images
		self.cfg = cfg
		self.drop_sensitive = drop_sensitive
		self.init_classifier()
		self.val_y_preds = torch.zeros((0,), dtype=torch.float32)
		self.val_y_trues = torch.zeros((0,), dtype=torch.int32)
		self.val_a = None

	def init_classifier(self):
		if self.images:
			network = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
			classifier = adapt_for_finetuning(network)
		else:
			classifier = NeuralNetwork(input_features=self.input_features,
										output_features=1,
										hidden_features=self.cfg['hidden_features'],
										nb_hidden_layers=self.cfg['nb_hidden_layers'],
										drop_sensitive=self.drop_sensitive,
										random_state=self.cfg.seed,
										a_idx=self.sensitive_index,
										dropout=self.cfg['dropout'])  # input_features?
		self.set_classifier(classifier)

	def set_classifier(self, classifier):
		self.classifier = classifier

	def forward(self, x):
		return self.classifier(x.to(self.cfg.device))

	def training_step(self, batch, batch_idx):
		_, X, y = batch
		y_pred = self.forward(X).flatten()
		y, y_pred = y.cpu(), y_pred.cpu()
		loss = self.loss(y_pred, y)
		acc = self.acc(y_pred, y)
		self.log('train/loss', loss, prog_bar=True)
		self.log('train/acc', acc, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		a, X, y_true = batch
		y_pred = self.forward(X).flatten()
		self.val_y_preds = torch.concatenate([self.val_y_preds, y_pred.cpu()])
		self.val_y_trues = torch.concatenate([self.val_y_trues, y_true.cpu()])
		if len(a.shape) == 1:
			a = a.unsqueeze(1)
		if self.val_a is None:
			self.val_a = a.cpu()
		else:
			self.val_a = torch.concatenate([self.val_a, a.cpu()])

	def on_validation_epoch_end(self) -> None:
		y_pred = self.val_y_preds
		y_true = self.val_y_trues
		a = self.val_a[:, 0]
		self.val_y_preds = torch.zeros((0,), dtype=torch.float32)
		self.val_y_trues = torch.zeros((0,), dtype=torch.int32)
		self.val_a = None

		metrics_X = get_all_metrics(*make_standard_from_torch(a, (y_pred > torch.tensor(0.5)).int(), y_true))

		self.log('val/acc', self.acc(y_pred, y_true))
		self.log('val/f1', self.f1(y_pred, y_true))
		self.log('val/loss', self.loss(y_pred, y_true))
		self.log('val/auroc', self.auroc(y_pred, y_true))
		self.log('val/avg_pr', self.ap(y_pred, y_true))
		self.log('val/priv_mean_pred', torch.mean(y_pred[a == 0]))
		self.log('val/unpriv_mean_pred', torch.mean(y_pred[a != 0]))
		self.log('val/brier', self.brier(y_pred, y_true))
		self.log('val/ece', self.ece(y_pred, y_true))
		self.log('val/psd', probabilistic_statistical_disparity(y_pred, a))
		self.log('val/eq_odds', probabilistic_equalized_odds_disparity(y_pred, y_true, a))
		self.log('val/eq_opp', probabilistic_equalized_opportunity_disparity(y_pred, y_true, a))
		self.log('val/statistical_disparity', metrics_X['Statistical Disparity'])
		self.log('val/priv_pos_rate', metrics_X['Privileged Positive Rate'])
		self.log('val/unpriv_pos_rate', metrics_X['Unprivileged Positive Rate'])
		self.log('val/TPR_diff', metrics_X['TPR_diff'])
		self.log('val/FPR_diff', metrics_X['FPR_diff'])

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		return optimizer

class LightningDPL(MyLightningModule):
	def __init__(self, cfg, images=False, input_features=5, pos_weight=1, fair_params=None):
		self.fair_params_flag = fair_params is not None
		self.fair_params = fair_params
		self.images = images
		super().__init__(cfg, images, input_features, cfg.drop_sensitive, pos_weight)

		self.use_logic_flag = True
		self.module = build_module(self.classifier, cfg, images=images)

		if fair_params: # when use_logic_flag is False still use logic but with the fair_params (in case of "bias drift")
			self.fair_module = build_module(self.classifier, fair_params, images=images)

	def set_classifier(self, classifier):
		self.classifier = classifier
		self.module = build_module(self.classifier, self.cfg, images=self.images)
		if self.fair_params_flag:
			self.fair_module = build_module(self.classifier, self.fair_params, images=self.images)

	def use_logic(self, mode: bool = True):
		self.use_logic_flag = mode

	# overrides MyLightningModule.forward
	def forward(self, X, a) -> torch.Tensor:
		X, a = X.to(self.cfg.device), a.to(self.cfg.device)
		if self.use_logic_flag:
			return self.module(X, a)
		elif self.fair_params_flag:
			return self.fair_module(X, a)
		return self.classifier(X)

	# overrides MyLightningModule.training_step
	def training_step(self, batch, batch_idx):
		a, X, y = batch
		y_pred = self.forward(X, a).flatten()
		y, y_pred = y.cpu(), y_pred.cpu()
		loss = self.loss(y_pred, y)
		acc = self.acc(y_pred, y)
		self.log('train/loss', loss, prog_bar=True)
		self.log('train/acc', acc, prog_bar=True)

		return loss

	# overrides MyLightningModule.validation_step
	def validation_step(self, batch, batch_idx):
		a, X, y_true = batch
		y_pred = self.forward(X, a).flatten()
		self.val_y_preds = torch.concatenate([self.val_y_preds, y_pred.cpu()])
		self.val_y_trues = torch.concatenate([self.val_y_trues, y_true.cpu()])
		if len(a.shape) == 1:
			a = a.unsqueeze(1)
		if self.val_a is None:
			self.val_a = a.cpu()
		else:
			self.val_a = torch.concatenate([self.val_a, a.cpu()])

class LightningNN(MyLightningModule):
	def __init__(self, cfg, images=False, input_features=5, drop_sensitive=False, pos_weight=1):
		super().__init__(cfg, images, input_features, drop_sensitive, pos_weight)

class NeuralNetwork(nn.Module):
	def __init__(self, input_features=5, output_features=1, hidden_features=None, nb_hidden_layers=None, drop_sensitive=False, random_state=None, a_idx=0, dropout=None):
		if nb_hidden_layers is None: nb_hidden_layers = 2
		if hidden_features is None: hidden_features = 100
		if dropout is None: dropout = 0
		self.drop_sensitive = drop_sensitive
		self.a_idx = a_idx
		input_features = input_features - 1 if drop_sensitive else input_features
		super().__init__()
		if random_state is not None:
			torch.manual_seed(random_state)
		self.fc1 = nn.Linear(input_features, hidden_features)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_features, output_features)
		self.activation = nn.Sigmoid()
		self.hidden_layers = nn.ModuleList()
		self.dropout = nn.Dropout(p=dropout)
		for _ in range(nb_hidden_layers):
			self.hidden_layers.append(nn.Linear(hidden_features, hidden_features))

	def forward(self, x):
		if self.drop_sensitive:
			x = torch.concatenate((x[:, :self.a_idx], x[:, self.a_idx+1:]), dim=1)  # drop sensitive feature
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		for hidden_layer in self.hidden_layers:
			x = hidden_layer(x)
			x = self.relu(x)
			x = self.dropout(x)
		x = self.fc2(x)
		x = self.activation(x)
		return x