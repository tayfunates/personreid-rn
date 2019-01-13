from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
	"""
	Randomly sample N identities, then for each identity,
	randomly sample K instances, therefore batch size is N*K.

	Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

	Args:
		data_source (Dataset): dataset to sample from.
		num_instances (int): number of instances per identity.
	"""
	def __init__(self, data_source, num_instances=4):
		self.data_source = data_source
		self.num_instances = num_instances
		self.index_dic = defaultdict(list)
		for index, (_, pid, _) in enumerate(data_source):
			self.index_dic[pid].append(index)
		self.pids = list(self.index_dic.keys())
		self.num_identities = len(self.pids)

	def __iter__(self):
		indices = torch.randperm(self.num_identities)
		ret = []
		for i in indices:
			pid = self.pids[i]
			t = self.index_dic[pid]
			replace = False if len(t) >= self.num_instances else True
			t = np.random.choice(t, size=self.num_instances, replace=replace)
			ret.extend(t)
		return iter(ret)

	def __len__(self):
		return self.num_identities * self.num_instances
	
	
class RandomPairwiseSampler(Sampler):
	
	def __init__(self, data_source, batch_size):
		self.data_source = data_source
		self.num_of_pos = int(batch_size/2)
		self.num_of_neg = batch_size - self.num_of_pos
		self.index_dic = defaultdict(list)
		self.batch_size = batch_size
		np.random.seed(123)
		for index, (_, pid, _) in enumerate(data_source):
			self.index_dic[pid].append(index)
		
		self.pids = list(self.index_dic.keys())
		self.num_identities = len(self.pids)
	
	def __iter__(self):
		indices = np.random.permutation(self.num_identities).tolist()
		ret = []
		for i in indices:
			pid = self.pids[i]
			pos_t = self.index_dic[pid]
			replace = False if len(pos_t) >= self.num_of_pos else True
			pos_t = np.random.choice(pos_t, size=self.num_of_pos, replace=replace).tolist()
			
			neg_t = []
			neg_indices = np.random.permutation(self.num_identities).tolist()
			neg_indices.remove(i)
			neg_indices = np.random.choice(neg_indices, size=self.num_of_neg, replace=False)
			for j in neg_indices:
				neg_pid = self.pids[j]
				neg_idxs = self.index_dic[neg_pid]
				neg_id = np.random.choice(neg_idxs, size=1, replace=False).tolist()
				neg_t.extend(neg_id)
			
			ret.extend(pos_t)
			ret.extend(neg_t)

		return iter(ret)

	def __len__(self):
		return self.batch_size * self.num_identities
    
        
        
        
        
        
        