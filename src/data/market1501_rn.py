from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import pickle
import random
import os.path as osp
from scipy.io import loadmat
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))

from utility.utils import mkdir_if_missing, write_json, read_json

random.seed(123)

"""Dataset classes"""
class Market1501(object):
	"""
	Market1501

	Reference:
	Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
	
	Dataset statistics:
	# identities: 1501 (+1 for background)
	# images: 12936 (train) + 3368 (query) + 15913 (gallery) 
	"""
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	root = osp.join(cur_dir, '..', '..', 'outputs', 'market1501')
	train_path = osp.join(root, 'train_features.pickle')
	query_path = osp.join(root, 'query_features.pickle')
	gallery_path = osp.join(root, 'gallery_features.pickle')
	val_id_count = 100
	
	def __init__(self, **kwargs):
		train, num_train_pids, num_train_imgs, self.train_features = self._create_dataset(self.train_path, relabel=True)
		query, num_query_pids, num_query_imgs, self.query_features = self._create_dataset(self.query_path, relabel=False)
		gallery, num_gallery_pids, num_gallery_imgs, self.gallery_features = self._create_dataset(self.gallery_path, relabel=False)
		val_query, val_gallery = self.split_test_val(query, gallery)
		#train, val_query, val_gallery = self.split_training(query, gallery)

		num_train_pids = num_train_pids
		num_total_pids = num_train_pids + num_query_pids
		num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

		print("=> Market1501 loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  subset   | # ids | # images")
		print("  ------------------------------")
		print("  train       | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
		print("  val query   | {:5d} | {:8d}".format(self.val_id_count, len(val_query)))
		print("  val gallery | {:5d} | {:8d}".format(self.val_id_count, len(val_gallery)))
		print("  query       | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
		print("  gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
		print("  ---------------------------------")
		print("  total       | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
		print("  ------------------------------")

		self.train = train
		self.query = query
		self.gallery = gallery
		self.val_query = val_query
		self.val_gallery = val_gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

	def _create_dataset(self, path, relabel=True):
		with open(path, "r") as handle:
			_set = pickle.load(handle)
			features = _set["features"]

		dataset = []
		for img_path in _set["features"]:
			pid = _set["pids"][img_path]
			camid = _set["camids"][img_path]
			dataset.append((img_path, int(pid), int(camid)))

		num_pids = len(set([_set["pids"][i][0] for i in _set["pids"]]))
		num_imgs = len(dataset)
		return dataset, num_pids, num_imgs, features
	
	def split_test_val(self, query, gallery):
		val_query = []
		val_gallery = []
		
		pids = set()
		for (img_path, pid, camid) in query:
			pids.add(pid)

		val_ids = random.sample(pids, self.val_id_count)
		
		for img_path, pid, camid in query:
			if pid in val_ids:
				val_query.append((img_path, pid, camid))
		
		for img_path, pid, camid in gallery:
			if pid in val_ids:
				val_gallery.append((img_path, pid, camid)) 
		
		return val_query, val_gallery
