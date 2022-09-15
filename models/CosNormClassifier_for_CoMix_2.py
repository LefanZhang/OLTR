import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

import pdb

class CosNorm_Classifier_CoMix(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier_CoMix, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = Parameter(torch.tensor([scale]).cuda())
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        #ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ex = input / norm_x
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())

def create_model(feat_dim, num_classes=1000, scale=16, stage1_weights=False, dataset=None, test=False, *args):
    print('Loading Cosine Norm Classifier For CoMix.')
    return CosNorm_Classifier_CoMix(in_dims=feat_dim, out_dims=num_classes, scale=scale)