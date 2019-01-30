from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
	
	
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
    
        
        
        
        
        
        