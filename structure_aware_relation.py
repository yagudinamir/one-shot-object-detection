import torch
import torch.nn
import torch.nn.functional as F
from torchvision.ops import roi_align, roi_pool
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import numpy as np

import time


class Predictor(nn.Module):

	def __init__(self, c1, c2, c3, h1, h2, h3):
		super(Predictor, self).__init__()
		self.num_classes = 1
		self.conv1 = nn.Conv2d(c1, c2, kernel_size=1)
		self.bn1 = nn.BatchNorm2d(c2)
		self.conv2 = nn.Conv2d(c2, c3, kernel_size=1)
		self.bn2 = nn.BatchNorm2d(c3)

		self.hidden1 = nn.Linear(h1, h2)
		self.bn3 = nn.BatchNorm1d(h2)
		self.hidden2 = nn.Linear(h2, h3)
		self.bn4 = nn.BatchNorm1d(h3)

		self.cls_score = nn.Linear(h3, self.num_classes)
		self.bbox_pred = nn.Linear(h3, self.num_classes * 4)

	def forward(self, features):
		features = F.relu(self.conv1(features))
		features = self.bn1(features)
		features = F.relu(self.conv2(features))
		features = self.bn2(features)
		features = features.flatten(start_dim=1)
		features = F.relu(self.hidden1(features))
		features = self.bn3(features)
		features = F.relu(self.hidden2(features))
		features = self.bn4(features)

		scores = self.cls_score(features)
		bbox_deltas = self.bbox_pred(features)

		cls_probs = nn.Sigmoid()(scores)
		return cls_probs, bbox_deltas


class StructureAwareRelationModule(nn.Module):

	def __init__(self, predictor, feature_map_size=(5, 5)):
		super(StructureAwareRelationModule, self).__init__()
		self.feature_map_size = feature_map_size
		self.predictor = predictor


	def forward(self, support, query, boxes):
		"""
		query (N, C, H, W)
		support (N, C, H, W)
		boxes (N, L_i, 4)
		"""
		assert query.shape[0] == support.shape[0] == boxes.shape[0]
		batch_size = query.shape[0]
		pooled_query = roi_align(query, boxes, self.feature_map_size)
		support_boxes = torch.LongTensor([[[0, 0, support.shape[2], support.shape[3]] for _ in range(image_boxes.shape[0])] for image_boxes in boxes])
		pooled_support = roi_align(support, support_boxes, self.feature_map_size)

		# pooled (K, C, self.feature_map_size[0], self.feature_map_size[1]])
		features = torch.cat((pooled_query, pooled_support), dim=1)
		return self.predictor(features)


def get_relation_loss(support, query, gt_probs, gt_bbox_deltas):
	predictor = Predictor(c1, c2, c3, h1, h2, h3)
	relation_model = StructureAwareRelationModule(predictor)

	cls_probs, bbox_deltas = relation_model(support, query)

	cls_loss = nn.BCELoss()(cls_probs, gt_probs)

	bbox_deltas, gt_bbox_deltas = bbox_deltas.flatten(), gt_bbox_deltas.flatten()
	bbox_loss = nn.MSELoss()(bbox_deltas, gt_bbox_deltas)

	return cls_loss, bbox_loss



