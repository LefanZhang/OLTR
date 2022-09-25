import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F

import pdb

class PrototypeUpdateLoss(nn.Module):
    def __init__(self, prototypes, s, sample_per_class):
        super(PrototypeUpdateLoss, self).__init__()
        # print(prototypes)
        self.num_classes, self.prototypes_num, self.feat_dim = prototypes.shape
        self.prototypes = nn.Parameter(prototypes.detach())
        self.s = nn.Parameter(torch.tensor(s).cuda())
        self.sample_per_class = sample_per_class.detach()

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        feat = F.normalize(feat, dim=1).detach()
        label = label.detach()
        normed_prototypes = F.normalize(self.prototypes, dim=2)
        mask = torch.zeros(batch_size, self.num_classes).to(device).scatter_(1, label.view(-1, 1), 1) # (batch_size, num_classes)
        weight = (torch.reciprocal(self.sample_per_class).view(1, -1).repeat(batch_size, 1) * mask).sum(dim=1)  # (batch_size)
        weight *= max(self.sample_per_class)
        # weight *= min(self.sample_per_class)

        logits, _ = normed_prototypes.matmul(feat.T).permute(2, 0, 1).max(dim=2)    # (batch_size, num_classes)
        s_logits = logits * self.s

        probs = (F.softmax(s_logits, dim=1)*mask).sum(dim=1)    # (batch_size)
        MSELoss = nn.MSELoss(reduction='none')
        loss = MSELoss(probs, torch.ones_like(probs))
        # print(loss, weight)
        loss = (loss * weight).mean()

        return loss

    
def create_loss (s, prototypes, sample_per_class):
    print('Loading Prototype Update Loss.')
    return PrototypeUpdateLoss(prototypes, s, sample_per_class)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feat_dim = 10
    num_classes = 100
    prototypes_num = 4
    prototypes = torch.randn(num_classes, prototypes_num, feat_dim).to(device)
    s = 16.
    sample_per_class = torch.arange(num_classes).to(device) + 1
    print(sample_per_class)
    pul = create_loss(s, prototypes, sample_per_class)
    
    batch_size = 16
    feat = torch.randn(batch_size, feat_dim).to(device)
    label = torch.arange(16).view(-1).to(device)
    loss = pul(feat, label)
    print(loss)