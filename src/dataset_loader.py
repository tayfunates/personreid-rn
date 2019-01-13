from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

def read_image(img_path):
	"""Keep reading image until succeed.
	This can avoid IOError incurred by heavy IO process."""
	got_img = False
	while not got_img:
		try:
			img = Image.open(img_path).convert('RGB')
			got_img = True
		except IOError:
			print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
			pass
	return img

class ImageDataset(Dataset):
	"""Image Person ReID Dataset"""
	def __init__(self, dataset, include_path=False, transform=None):
		self.dataset = dataset
		self.transform = transform
		self.include_path = include_path
		self.samples = dict()
	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img_path, pid, camid = self.dataset[index]
		if img_path in self.samples:
			return img_path, torch.zeros((3,256,128)), pid, camid
			
		self.samples[img_path] = True
		img = read_image(img_path)
		if self.transform is not None:
			img = self.transform(img)
		if self.include_path:
			return img_path, img, pid, camid
		return img, pid, camid
	
class FeatureDataset(Dataset):
	"""Features ReID Dataset"""
	def __init__(self, dataset, features):
		self.dataset = dataset
		self.features = features

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img_path, pid, camid = self.dataset[index]
		feature = self.features[img_path]
		return torch.from_numpy(feature), torch.tensor(pid), torch.tensor(camid)






