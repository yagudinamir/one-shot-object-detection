import torch
import torch.nn as nn
import torch.nn.functional as F
from structure_aware_relation import Predictor, StructureAwareRelationModule, get_relation_loss


def test_shapes(info):
	features = torch.randn(2, 3, 5, 5)
	predictor = Predictor(info)
	cls_probs, bbox_deltas = predictor(features)

	assert cls_probs.shape == (2, 1)
	assert bbox_deltas.shape == (2, 4)

	support = torch.randn(2, 7, 11, 17)
	query = torch.randn(2, 7, 61, 47)
	boxes = [torch.FloatTensor([[0, 0, 11, 17], [0, 0, 11, 17]]), torch.FloatTensor([[0, 2, 11, 19]])]
	gt_probs = torch.LongTensor([0, 1])
	gt_bbox_deltas = torch.FloatTensor([[2, 2, 2, 2], [3, 3, 3, 3]])

	cls_loss, bbox_loss = get_relation_loss(support, query, boxes, gt_probs, gt_bbox_deltas, info)

	assert cls_probs.shape == (2, 1)
	assert bbox_deltas.shape == (2, 4)








if __name__ == '__main__':
	test_shapes({
		'c1': 3, 
		'c2': 32,
		'c3': 16,
		'h1': 16 * 5 * 5,
		'h2': 128,
		'h3': 32,
	})