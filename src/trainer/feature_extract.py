from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

sys.path.append("..")

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data.data_manager as data_manager
from data.dataset_loader import ImageDataset
import utility.transforms as T
import models
from utility.losses import CrossEntropyLabelSmooth, DeepSupervision
from utility.utils import AverageMeter, Logger, save_checkpoint
from utility.eval_metrics import evaluate
from models.optimizers import init_optim
import pickle

gpu_devices = "0"
save_dir = 'log'
seed = 1
root = "data"
height = 256
width = 128

train_batch = 1
test_batch = 1
workers = 1
arch = "resnet50rn"
optim = "adam"
lr = 0.001
weight_decay = 5e-04
gamma = 0.1
decay_stepsize = 20
start_epoch = 0
resume = "../../weights/resnet50_xent_market1501.pth.tar"
evaluate_only = False
max_epoch = 180
eval_step = 2
print_freq = 1
use_metric_cuhk03 = False
extract_features = True

dataset_name = "market1501"
current_path = os.path.dirname(os.path.realpath(__file__))
train_save_path = os.path.join(current_path, "..", "outputs", "market1501", "train_features.pickle" )
query_save_path = os.path.join(current_path, "..", "outputs", "market1501", "query_features.pickle" )
gallery_save_path = os.path.join(current_path, "..", "outputs", "market1501", "gallery_features.pickle" )

def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
	losses = AverageMeter()
	batch_time = AverageMeter()
	data_time = AverageMeter()

	model.train()

	end = time.time()
	for batch_idx, (imgs, pids, _) in enumerate(trainloader):
		# measure data loading time
		data_time.update(time.time() - end)

		if use_gpu:
			imgs, pids = imgs.cuda(), pids.cuda()
		outputs = model(imgs)
		if isinstance(outputs, tuple):
			loss = DeepSupervision(criterion, outputs, pids)
		else:
			loss = criterion(outputs, pids)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), pids.size(0))

		if (batch_idx+1) % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
				   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,
				   data_time=data_time, loss=losses))

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
	batch_time = AverageMeter()
	
	model.eval()

	with torch.no_grad():
		qf, q_pids, q_camids = [], [], []
		for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
			if use_gpu: imgs = imgs.cuda()

			end = time.time()
			features = model(imgs)
			batch_time.update(time.time() - end)
			
			features = features.data.cpu()
			qf.append(features)
			q_pids.extend(pids)
			q_camids.extend(camids)
		qf = torch.cat(qf, 0)
		q_pids = np.asarray(q_pids)
		q_camids = np.asarray(q_camids)

		print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

		gf, g_pids, g_camids = [], [], []
		end = time.time()
		for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
			if use_gpu: imgs = imgs.cuda()

			end = time.time()
			features = model(imgs)
			batch_time.update(time.time() - end)

			features = features.data.cpu()
			gf.append(features)
			g_pids.extend(pids)
			g_camids.extend(camids)
		gf = torch.cat(gf, 0)
		g_pids = np.asarray(g_pids)
		g_camids = np.asarray(g_camids)

		print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

	print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
	
	m, n = qf.size(0), gf.size(0)
	distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
			  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
	distmat.addmm_(1, -2, qf, gf.t())
	distmat = distmat.numpy()

	print("Computing CMC and mAP")
	cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=use_metric_cuhk03)

	print("Results ----------")
	print("mAP: {:.1%}".format(mAP))
	print("CMC curve")
	for r in ranks:
		print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
	print("------------------")

	return cmc[0]


def feature_extract(model, dataloader, use_gpu):
	batch_time = AverageMeter()
	model.eval()

	with torch.no_grad():
		qf, q_pids, q_camids = {}, {}, {}
		for batch_idx, (img_path, imgs, pids, camids) in enumerate(dataloader):
			print(batch_idx, len(dataloader))
			if use_gpu: imgs = imgs.cuda()

			end = time.time()
			features = model(imgs)
			batch_time.update(time.time() - end)
			
			features = features.data.cpu().numpy()
			qf[img_path[0]] = features
			q_pids[img_path[0]] = pids.data.numpy()
			q_camids[img_path[0]] = camids.data.numpy()

		print("Extracted features for set, obtained {}-by-{} matrix".format(len(qf), qf[qf.keys()[0]].shape))
	return qf, q_pids, q_camids

if __name__ == '__main__':
	torch.manual_seed(seed)
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices 
	use_gpu = torch.cuda.is_available()

	sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))

	print("Currently using GPU {}".format(gpu_devices))
	cudnn.benchmark = True
	torch.cuda.manual_seed_all(seed)


	print("Initializing dataset {}".format(dataset_name))
	dataset = data_manager.init_img_dataset(root=root, name=dataset_name)

	transform_train = T.Compose([
		T.Random2DTranslation(height, width),
		T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	transform_test = T.Compose([
		T.Resize((height, width)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	pin_memory = True if use_gpu else False

	trainloader = DataLoader(
		ImageDataset(dataset.train, include_path=extract_features, transform=transform_train),
		batch_size=train_batch, shuffle=not extract_features, num_workers=workers,
		pin_memory=pin_memory, drop_last=True,
	)

	queryloader = DataLoader(
		ImageDataset(dataset.query, include_path=extract_features, transform=transform_test),
		batch_size=test_batch, shuffle=False, num_workers=workers,
		pin_memory=pin_memory, drop_last=False,
	)

	galleryloader = DataLoader(
		ImageDataset(dataset.gallery, include_path=extract_features, transform=transform_test),
		batch_size=test_batch, shuffle=False, num_workers=workers,
		pin_memory=pin_memory, drop_last=False,
	)

	print("Initializing model: {}".format(arch))
	model = models.init_model(name=arch, num_classes=dataset.num_train_pids, loss=None, use_gpu=use_gpu)
	model.freeze_base()
	model.cuda()
	
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

	criterion = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
	optimizer = init_optim(optim, model.classifier.parameters(), lr, weight_decay)
	if decay_stepsize > 0:
		scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_stepsize, gamma=gamma)

	if resume:
		print("Loading checkpoint from '{}'".format(resume))
		checkpoint = torch.load(resume)
		model.load_state_dict(checkpoint['state_dict'])
		start_epoch = checkpoint['epoch']

	if evaluate_only:
		print("Evaluate only")
		test(model, queryloader, galleryloader, use_gpu)
		exit
	
	if extract_features:
		train_feature, train_pids, train_camids = feature_extract(model, trainloader, use_gpu)
		query_feature, query_pids, query_camids = feature_extract(model, queryloader, use_gpu)
		gallery_feature, gallery_pids, gallery_camids = feature_extract(model, galleryloader, use_gpu)

		train_dict = {}
		train_dict["features"] = train_feature
		train_dict["pids"] = train_pids
		train_dict["camids"] = train_camids
		
		query_dict = {}
		query_dict["features"] = query_feature
		query_dict["pids"] = query_pids
		query_dict["camids"] = query_camids
		
		gallery_dict = {}
		gallery_dict["features"] = gallery_feature
		gallery_dict["pids"] = gallery_pids
		gallery_dict["camids"] = gallery_camids
		
		with open(train_save_path, "w") as handle:
			pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		with open(query_save_path, "w") as handle:
			pickle.dump(query_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		with open(gallery_save_path, "w") as handle:
			pickle.dump(gallery_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		exit

	start_epoch = 0
	start_time = time.time()
	train_time = 0
	best_rank1 = -np.inf
	best_epoch = 0
	print("==> Start training")

	for epoch in range(start_epoch, max_epoch):
		start_train_time = time.time()
		train(epoch, model, criterion, optimizer, trainloader, use_gpu)
		train_time += round(time.time() - start_train_time)
		
		if decay_stepsize> 0: scheduler.step()
		
		if eval_step > 0 and (epoch+1) % eval_step == 0 or (epoch+1) == max_epoch:
			print("==> Test")
			rank1 = test(model, queryloader, galleryloader, use_gpu)
			is_best = rank1 > best_rank1
			if is_best:
				best_rank1 = rank1
				best_epoch = epoch + 1

			state_dict = model.state_dict()
			save_checkpoint({
				'state_dict': state_dict,
				'rank1': rank1,
				'epoch': epoch,
			}, is_best, "syntetic_softmax_scratch_", osp.join(save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

	print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

	elapsed = round(time.time() - start_time)
	elapsed = str(datetime.timedelta(seconds=elapsed))
	train_time = str(datetime.timedelta(seconds=train_time))
	print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
