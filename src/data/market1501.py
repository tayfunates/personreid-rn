from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))

from utility.utils import mkdir_if_missing, write_json, read_json

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
	f_root = osp.join(cur_dir, '..', 'outputs', 'market1501')
	f_train_path = osp.join(f_root, 'train_features.pickle')
	f_query_path = osp.join(f_root, 'query_features.pickle')
	f_gallery_path = osp.join(f_root, 'gallery_features.pickle')
	
	root = '../../data/market1501'
	train_dir = osp.join(root, 'bounding_box_train')
	query_dir = osp.join(root, 'query')
	gallery_dir = osp.join(root, 'bounding_box_test')
	val_id_count = 0
	val_query_instances = 3

	def __init__(self, **kwargs):
		self._check_before_run()

		train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
		train, val_query, val_gallery = self.split_training(train)
		query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
		gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
		num_train_pids = num_train_pids - self.val_id_count
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

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.root):
			raise RuntimeError("'{}' is not available".format(self.root))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.query_dir):
			raise RuntimeError("'{}' is not available".format(self.query_dir))
		if not osp.exists(self.gallery_dir):
			raise RuntimeError("'{}' is not available".format(self.gallery_dir))

	def _process_dir(self, dir_path, relabel=False):
		img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
		pattern = re.compile(r'([-\d]+)_c(\d)')

		pid_container = set()
		for img_path in img_paths:
			pid, _ = map(int, pattern.search(img_path).groups())
			if pid == -1: continue  # junk images are just ignored
			pid_container.add(pid)
		pid2label = {pid:label for label, pid in enumerate(pid_container)}

		dataset = []
		for img_path in img_paths:
			pid, camid = map(int, pattern.search(img_path).groups())
			if pid == -1: continue  # junk images are just ignored
			assert 0 <= pid <= 1501  # pid == 0 means background
			assert 1 <= camid <= 6
			camid -= 1 # index starts from 0
			if relabel: pid = pid2label[pid]
			dataset.append((img_path, pid, camid))

		num_pids = len(pid_container)
		num_imgs = len(dataset)
		return dataset, num_pids, num_imgs

	def split_training(self, dataset):
		id_dict = dict()
		val_query = []
		val_gallery = []
		for (img_path, pid, camid) in dataset:
			if pid not in id_dict:
				id_dict[pid] = []
			id_dict[pid].append((img_path, pid, camid))
			
		val_ids = id_dict.keys()
		val_ids.sort(reverse=True)
		val_ids = val_ids[:self.val_id_count]
		pid2label = {pid:label for label, pid in enumerate(val_ids)}
		train_set = []
		
		for val_pid in val_ids:
			q_count = 0
			for (img_path, pid, camid) in id_dict[val_pid]:
				if q_count < self.val_query_instances:
					val_query.append((img_path, pid2label[pid], camid))
				else:
					val_gallery.append((img_path, pid2label[pid], camid))
				q_count += 1
		
		for pid in id_dict:
			if pid not in val_ids:
				train_set.extend(id_dict[pid])
		
		return train_set, val_query, val_gallery
			