import torch
import torch.nn as nn
import torch.nn.functional as F
from structure_aware_relation import Predictor, StructureAwareRelationModule, get_relation_loss
from fcos.model.fcos import FCOSDetector, FCOS
from fcos.model.loss import GenTargets


def test_fcos():
    b, c, h, w = 7, 3, 480, 320
    hs, ws = 160, 220
    model = FCOS()
    query = torch.randn(b, c, h, w)
    support = torch.randn(b, c, hs, ws)
    m = 3
    boxes = [[5, 5, 20, 22], [3, 5, 32, 11], [10, 10, 20, 20]]
    classes = [0, 1, 0]
    boxes = torch.FloatTensor([boxes for _ in range(b)]) * 10
    classes = torch.LongTensor([classes for _ in range(b)])
    print(classes.shape, boxes.shape)
    detector = FCOSDetector()
    losses, out, targets, features = detector([query, support, boxes, classes])
    cls_targets, cnt_targets, reg_targets = targets
    print(reg_targets.shape, cls_targets.shape)
    print(len(out[0]), len(out[1]), len(features[0]))

if __name__ == '__main__':
    test_fcos()