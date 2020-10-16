import torch
import torch.nn as nn
from fcos.model.fcos import FCOSDetector
from structure_aware_relation import StructureAwareRelationModule, Predictor


class OneShotDetector(nn.Module):
    def __init__(self, info, config=None):
        super().__init__()
        self.first_stage = FCOSDetector(config=config)
        self.predictor = Predictor(info)
        self.second_stage = StructureAwareRelationModule(predictor=self.predictor)

    def forward(self, inputs):
        '''
        inputs: list [batch_queries, batch_support, batch_boxes, batch_classes]
        '''
        strides = [8, 16, 32, 64, 128]  # TODO: use config
        first_stage_losses, out, _, features = self.first_stage(inputs)
        for (_, _, boxes), (features_query, features_support) in zip(out, features):
            # boxes.shape == [batch_size,4,h,w]
            # Several questions
            pass
