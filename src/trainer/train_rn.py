from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data.market1501_rn import Market1501
import utility.transforms as T
from data.dataset_loader import FeatureDataset
from models.ReidRN import ReID_RN
from utility.utils import AverageMeter, Logger, save_checkpoint
from utility.eval_metrics import evaluate
from models.optimizers import init_optim
from data.samplers import RandomPairwiseSampler
#from utility.reidtools import visualize_ranked_results

gpu_devices = "0"
seed = 1
root = "data"
height = 256
width = 128

train_batch = 96 # sampling batch
query_batch = 1 	# sampling batch
gallery_batch = 128
val_query_batch = 1
val_gallery_batch = 16
workers = 0
optim = "adam"
lr = 0.00001
weight_decay = 5e-04
gamma = 0.5
decay_stepsize = 600
start_epoch = 0
dropout_prob = 0.2


# Model Parameters (5 different settings)
hiddim1 = [512, 512, 1024, 1024, 2048]
numhid1 = [3, 6, 3, 6, 6]
hiddim2 = [256, 256, 256, 512, 512]
numhid2 = [2, 3, 2, 3, 3]
modelNum = 2

# The one below is for discarding bg objects during training. If we have n * m pairs for object representations
# and if we want to discard bg objects, we train with (n - 2) * (m - 2) pairs. E.g If we have 32 objects for learning relations
# we only train with 12 objects by discarding the boundary objects
discardBGObjects = False

resume = "/home/burak/Desktop/workspace/Reid-RN/Reid-Relation/src/trainer/log/MarketRN/o84_lr0.00001_model2_dropout0.2_discardBGObjectsFalse/rn_from_pretrained_best_model.pth.tar"
evaluate_only = True

# EVALUATION ICIN BURAYI AC
# resume = osp.join(cur_dir, "log_pretrained", "rn_from_pretrained_best_model.pth.tar")
# resume = osp.join(cur_dir, "log", "MarketRN", "lr0.00005", "rn_from_pretrained_best_model.pth.tar")
#resume = "/home/burak/Desktop/workspace/Reid-RN/Reid-Relation/src/trainer/log/MarketRN/lr0.00001_model2_dropout0.2_discardBGObjectsFalse_rSize512/rn_from_pretrained_best_model.pth.tar"

max_epoch = 180
eval_step = 10
print_freq = 100
test_print_freq = 10
use_metric_cuhk03 = False
dist_alpha = torch.tensor(1.0).cuda()

logFolderName = "o84_lr{learning_rate:7.5f}_model{model_num}_dropout{dp}_discardBGObjects{dbg}".format(learning_rate=lr, model_num=modelNum, dp=dropout_prob, dbg=discardBGObjects)

save_dir = osp.join("log", "MarketRN", logFolderName)

while os.path.isdir(save_dir):
	save_dir = save_dir + "c"

if resume:
	save_dir = osp.dirname(resume)

def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
	losses = AverageMeter()
	batch_time = AverageMeter()
	data_time = AverageMeter()

	model.train()

	end = time.time()
	true_y = torch.tensor([[1.0]]).cuda()
	false_y = torch.tensor([[0.0]]).cuda()
	for batch_idx, (imgs, pids, camids) in enumerate(trainloader):
		# measure data loading time
		data_time.update(time.time() - end)

		if use_gpu:
			imgs, pids = imgs.cuda(), pids.cuda()
		
		length = imgs.shape[0]
		pos_size = int(length/2)
		pos_imgs = imgs[:pos_size]
		neg_imgs = imgs[pos_size:]
		pos_imgs_shifted = torch.cat((pos_imgs[1:],pos_imgs[:1]), 0)

		loss = 0

		x1 = torch.cat((pos_imgs, pos_imgs), 0).squeeze()
		x2 = torch.cat((pos_imgs_shifted, neg_imgs), 0).squeeze()
		if(discardBGObjects):
			x1 = x1[:, :, 1:x1.shape[2]-1, 1:x1.shape[3]-1].clone()
			x2 = x2[:, :, 1:x2.shape[2]-1, 1:x2.shape[3]-1].clone()

		ground_truths = torch.cuda.FloatTensor([1] * pos_size + [0] * pos_size).unsqueeze(1)

		outputs = model(x1, x2)

		loss = criterion(outputs, ground_truths)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.update(loss.item(), 1)
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		
		if (batch_idx+1) % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
				   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,
				   data_time=data_time, loss=losses))

def test(model, queryloader, galleryloader, use_gpu, galleryBatch, ranks=[1, 5, 10, 20]):
	batch_time = AverageMeter()
		
	model.eval()
	distmat = np.zeros((len(queryloader.dataset), len(galleryloader.dataset)))
	q_pids, q_camids = [], []
	g_pids, g_camids = [], []

	with torch.no_grad():
		for query_idx, (q_img, q_pid, q_camid) in enumerate(queryloader):
			if query_idx % test_print_freq == 0:
				print("Query index:", query_idx, "Length:", len(queryloader))

			# fill query_pids & query_camids
			q_pids.append(q_pid)
			q_camids.append(q_camid)
			end = time.time()
			q_img = q_img.cuda()
			q_img = q_img.squeeze(1)
			q_imgs = torch.cat([q_img]*galleryBatch, 0).squeeze()
			for gallery_idx, (g_img, g_pid, g_camid) in enumerate(galleryloader):
				# fill gallery_pids & gallery_camids once
				if query_idx == 0:
					g_pids.append(g_pid)
					g_camids.append(g_camid)

				if gallery_idx == len(galleryloader) - 1:
					q_imgs = torch.cat([q_img]*g_img.size(0), 0).squeeze()

				g_img = g_img.cuda()
				g_img = g_img.squeeze(1)

				y_pred = model(q_imgs, g_img)
				y_pred = y_pred.squeeze()
				y_pred_np = y_pred.data.cpu().numpy()

				start_idx = gallery_idx*galleryBatch
				end_idx = start_idx + galleryBatch
				distmat[query_idx, start_idx:end_idx] = 1.0-y_pred_np
			batch_time.update(time.time() - end)

	print('Evaluate: Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time))
	print("Computing CMC and mAP")	
	q_pids = np.asarray(q_pids)
	q_camids = np.asarray(q_camids)
	g_pids = np.hstack(g_pids)
	g_camids = np.hstack(g_camids)
	cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=use_metric_cuhk03)

	print("Results ----------")
	print("mAP: {:.1%}".format(mAP))
	print("CMC curve")
	for r in ranks:
		print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
	print("------------------")

	return cmc[0], distmat

if __name__ == '__main__':
	torch.manual_seed(seed)
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices 
	use_gpu = torch.cuda.is_available()

	if evaluate_only:
		sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))
	else:
		sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))


	print("Currently using GPU {}".format(gpu_devices))
	cudnn.benchmark = True
	torch.cuda.manual_seed_all(seed)


	print("Initializing dataset {}".format("Market1501RN"))
	dataset = Market1501()

	transform_train = T.Compose([
		T.Resize((height, width)),
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
		FeatureDataset(dataset.train, features=dataset.train_features),
		batch_size=train_batch, num_workers=workers,
		sampler=RandomPairwiseSampler(dataset.train, train_batch),
		pin_memory=pin_memory, drop_last=True,
	)

	queryloader = DataLoader(
		FeatureDataset(dataset.query, features=dataset.query_features),
		batch_size=query_batch, shuffle=False, num_workers=workers,
		pin_memory=pin_memory, drop_last=False,
	)

	galleryloader = DataLoader(
		FeatureDataset(dataset.gallery, features=dataset.gallery_features),
		batch_size=gallery_batch, shuffle=False, num_workers=workers,
		pin_memory=pin_memory, drop_last=False,
	)

	valQueryloader = DataLoader(
		FeatureDataset(dataset.val_query, features=dataset.query_features),
		batch_size=val_query_batch, shuffle=False, num_workers=workers,
		pin_memory=pin_memory, drop_last=False,
	)

	valGalleryloader = DataLoader(
		FeatureDataset(dataset.val_gallery, features=dataset.gallery_features),
		batch_size=val_gallery_batch, shuffle=False, num_workers=workers,
		pin_memory=pin_memory, drop_last=False,
	)

	print("Initializing model: Reid_RN")
	#model = ReID_RN((4096, 512, 3), (512, 1, 512, 256, 2))
	model = ReID_RN((4096, hiddim1[modelNum], numhid1[modelNum]), (hiddim1[modelNum], 1, hiddim1[modelNum], hiddim2[modelNum], numhid2[modelNum]), dropout_prob)
	model.cuda()
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

	criterion = torch.nn.BCELoss()
	optimizer = init_optim(optim, model.parameters(), lr, weight_decay)
	if decay_stepsize > 0:
		scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_stepsize, gamma=gamma)

	start_epoch = 0
	best_rank1 = -np.inf

	if resume:
		print("Loading checkpoint from '{}'".format(resume))
		checkpoint = torch.load(resume)
		model.load_state_dict(checkpoint['state_dict'])
		start_epoch = checkpoint['epoch']
		best_rank1 = checkpoint['rank1']

	if evaluate_only:
		print("Evaluate only")
		_, distmat = test(model, queryloader, galleryloader, use_gpu, gallery_batch)
		#visualize_ranked_results(
        #        distmat, dataset,
        #        save_dir=osp.join(save_dir, 'ranked_results'),
        #        topk=20,
        #    )
		sys.exit(0)

	best_epoch = start_epoch
	start_time = time.time()
	train_time = 0
	print("==> Start training")

	for epoch in range(start_epoch, max_epoch):
		start_train_time = time.time()
		train(epoch, model, criterion, optimizer, trainloader, use_gpu)
		train_time += round(time.time() - start_train_time)
		
		if decay_stepsize> 0: scheduler.step()
		
		if eval_step > 0 and (epoch+1) % eval_step == 0 or (epoch+1) == max_epoch:
			print("==> Test")
			rank1, distmat = test(model, valQueryloader, valGalleryloader, use_gpu, val_gallery_batch)
			is_best = rank1 >= best_rank1
			if is_best:
				best_rank1 = rank1
				best_epoch = epoch + 1

			state_dict = model.state_dict()
			save_checkpoint({
				'state_dict': state_dict,
				'rank1': rank1,
				'epoch': epoch,
			}, is_best, "rn_from_pretrained_", osp.join(save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

	print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

	elapsed = round(time.time() - start_time)
	elapsed = str(datetime.timedelta(seconds=elapsed))
	train_time = str(datetime.timedelta(seconds=train_time))
	print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

	print("Starting Full Test With Model Which Is Obtained From Last Epoch")
	test(model, queryloader, galleryloader, use_gpu, gallery_batch)
