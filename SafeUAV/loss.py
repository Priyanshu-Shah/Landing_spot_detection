import torch as tr
import torch.nn.functional as F
import numpy as np

def l2_loss(networkOutput, label, Lambda=0):
	res1 = tr.sum( (networkOutput - label)**2, dim=1).sum(dim=1)
	return tr.mean(res1)

def classification_loss(y_pred, y_true):
	# (240, 320, 3) => (153600, 3) and (240, 320, 1) => (153600, )
	y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
	y_true = y_true.view(-1)
	y_pred = F.log_softmax(y_pred, dim=-1)
	loss = F.nll_loss(y_pred, y_true)
	return loss

# Cache the values so they are not computed for all metrics
TP, TN, FP, FN = None, None, None, None
def getTPTNFPFN(y_pred_batch, y_true_batch, computeNewValues):
	global TP, TN, FP, FN
	if computeNewValues:
		predTrue = y_pred_batch == True
		labelTrue = y_true_batch == True
		predFalse = y_pred_batch == False
		labelFalse = y_true_batch == False

		TP = np.sum(np.logical_and(predTrue, labelTrue))
		FP = np.sum(np.logical_and(predTrue, labelFalse))
		FN = np.sum(np.logical_and(predFalse, labelTrue))
		TN = np.sum(np.logical_and(predFalse, labelFalse))
	return TP, TN, FP, FN

import numpy as np
from plotter import plotter, show_plots

def testDataset(reader, args):
	generator = reader.iterate("train", miniBatchSize=args.batch_size)
	steps = reader.getNumIterations("train", miniBatchSize=args.batch_size)
	from functools import partial

	for items in generator:
		datas, labels = items
		MB = datas.shape[0]

		for i in range(MB):
			data = datas[i]
			label = labels[i]
			totalItems = len(args.data_dims) + len(args.label_dims)
			plotter(data, args.data_dims, reader.hvnTransform, startingIndex=1, totalItems=totalItems)
			ix = len(args.data_dims) + 1
			plotter(label, args.label_dims, reader.hvnTransform, startingIndex=ix, totalItems=totalItems)
			show_plots()

def mIoUMetric(y_pred_batch, y_true_batch, **kwargs):
	def IoU(y_pred_batch, y_true_batch):
		TP, _, FP, FN = getTPTNFPFN(y_pred_batch, y_true_batch, computeNewValues=True)
		return TP / (TP + FP + FN + 1e-8)

	y_pred_batch = np.argmax(y_pred_batch, axis=-1)
	y_true_batch = y_true_batch[..., 0]
	res = [IoU((y_pred_batch == i), (y_true_batch == i)) for i in range(3)]
	return np.mean(np.array(res))

def metterMetric(y_pred_batch, y_true_batch, **kwargs):
	F = kwargs["reader"].maximums["depth"]
	y_pred_new = y_pred_batch * F
	y_true_new = y_true_batch * F
	return np.mean(np.abs(y_pred_new - y_true_new))

def precisionMetric(y_pred_batch, y_true_batch, **kwargs):
	def precision(y_pred_batch, y_true_batch):
		TP, _, FP, _ = getTPTNFPFN(y_pred_batch, y_true_batch, computeNewValues=True)
		return TP / (TP + FP + 1e-8)

	y_pred_batch = np.argmax(y_pred_batch, axis=-1)
	y_true_batch = y_true_batch[..., 0]
	res = [precision((y_pred_batch == i), (y_true_batch == i)) for i in range(3)]
	return np.mean(np.array(res))

def recallMetric(y_pred_batch, y_true_batch, **kwargs):
	def recall(y_pred_batch, y_true_batch):
		TP, _, _, FN = getTPTNFPFN(y_pred_batch, y_true_batch, computeNewValues=True)
		return TP / (TP + FN + 1e-8)

	y_pred_batch = np.argmax(y_pred_batch, axis=-1)
	y_true_batch = y_true_batch[..., 0]
	res = [recall((y_pred_batch == i), (y_true_batch == i)) for i in range(3)]
	return np.mean(np.array(res))

def accuracyMetric(y_pred_batch, y_true_batch, **kwargs):
	def recall(y_pred_batch, y_true_batch):
		TP, TN, FP, FN = getTPTNFPFN(y_pred_batch, y_true_batch, computeNewValues=True)
		return (TP + TN) / (TP + FP + TN + FN + 1e-8)

	y_pred_batch = np.argmax(y_pred_batch, axis=-1)
	y_true_batch = y_true_batch[..., 0]
	res = [recall((y_pred_batch == i), (y_true_batch == i)) for i in range(3)]
	return np.mean(np.array(res))
