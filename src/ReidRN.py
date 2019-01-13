# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import pickle


class ReID_RN(nn.Module):
	def __init__(self, config_mlp, config_clf, dropout_prob=0.2):
		super(ReID_RN, self).__init__()
		self.mlp = RN_MLP(config_mlp[0], config_mlp[1], config_mlp[2], dropout_prob)
		self.clf = RN_CLF(config_clf[0], config_clf[1], config_clf[2], config_clf[3], config_clf[4], dropout_prob)
		self.total_dim = config_clf[0]
		
	def forward(self, X1, X2):
		x1 = X1.view(X1.shape[0], X1.shape[1], -1)
		x2 = X2.view(X2.shape[0], X2.shape[1], -1)

		z = torch.cat((x1, x2),1)
		z = z.permute(0, 2, 1)

		out = self.mlp.forward(z)
		out = out.sum(1)
		out = self.clf.forward(out)
		return out
		
	
class RN_MLP(nn.Module):
	def __init__(self, D_in, D_h=256, n_hidden=4, dropout_prob=0.2):
		super(RN_MLP, self).__init__()
		self.D_in = D_in
		self.D_h = D_h
		self.n_hidden = n_hidden
		self.create_network(dropout_prob)
			
	def create_network(self, dropout_prob):
		blocks = []
		blocks.extend(_make_block(self.D_in, self.D_h))
		for i in range(self.n_hidden):
			blocks.extend(_make_block(self.D_h, self.D_h, dropout_prob))
		self.layers = nn.ModuleList(blocks)
	
	def forward(self, X):
		out = X
		for layer in self.layers:
			out = layer(out)
		return out
	
	
class RN_CLF(nn.Module):
	def __init__(self, D_in, D_out=1, D_h1=256, D_h2=29, n_hidden=3, dropout_prob=0.2):
		super(RN_CLF, self).__init__()
		self.D_in = D_in
		self.D_h1 = D_h1
		self.D_h2 = D_h2
		self.D_out = D_out
		self.n_hidden = n_hidden
		self.create_network(dropout_prob)
		
	def create_network(self, dropout_prob=0.2):
		blocks = []
		blocks.extend(_make_block(self.D_in, self.D_h1))
		for i in range(self.n_hidden):
			blocks.extend(_make_block(self.D_h1, self.D_h1))
		self.hidden_layers = nn.ModuleList(blocks)
		self.clf = nn.ModuleList(_make_classifier(self.D_h1, self.D_h2, self.D_out, dropout_prob))
		
	def forward(self, X):
		out = X
		for layer in self.hidden_layers:
			out = layer(out)
		
		for layer in self.clf:
			out = layer(out)
		return F.sigmoid(out)
		

def _make_block(D_in, D_out, ignore_norm=False, dropout_prob=0.2):
	linear = nn.Linear(D_in, D_out)
	torch.nn.init.xavier_uniform_(linear.weight)
	activation = nn.ReLU()
	dropout = nn.Dropout(p=dropout_prob)
	layers = [linear, activation, dropout]
	return layers

def _make_classifier(D_in, D_h, D_out, ignore_norm=False, dropout_prob=0.2):
	linear = nn.Linear(D_in, D_h)
	torch.nn.init.xavier_uniform_(linear.weight)
	activation = nn.ReLU()
	dropout = nn.Dropout(p=dropout_prob)
	linear_cls = nn.Linear(D_h, D_out)
	layers = [linear, activation, dropout, linear_cls]
	return layers

if __name__ == '__main__':
	
	with open("/media/nihattekeli/My Book/ONUR/Reid-Relation/src/outputs/market1501/query_features.pickle", "r") as handle:
		features = pickle.load(handle)
	nnet = ReID_RN((4096, 256, 4), (256, 1, 256, 29, 3))
	