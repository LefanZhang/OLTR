import torch.nn as nn
import torch
import torch.nn.functional as F

class DisPSCLoss(nn.Module):
    def __init__(self, temp=0.1, eps=1e-3, weighted=False):
        super(DisPSCLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        self.weighted = weighted
        # print(self.eps)
    
    def forward(self, feat, label, prototypes, probs, sample_per_class=None, discriminative=False, balanced=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        feat = F.normalize(feat, dim=1) # normalize to 1
        batch_size, feat_dim = feat.shape
        num_classes, k, _ = prototypes.shape    # (num_classes, k, feat_dim)
        mask = torch.zeros(batch_size, num_classes).to(device).scatter_(1, label.view(-1, 1), 1) # (batch_size, num_classes)
        # print(mask)
        # print(mask.shape)

        if discriminative:
            # eps = 1e-3  # default 1e-3
            GT_probs = (probs*mask).sum(dim=1).view(-1, 1).repeat(1, num_classes)
            # print(GT_probs)
            dis_mask = torch.where(probs >= GT_probs, 1, 0)
            # print(dis_mask)
            # print(modified_probs)
            # weight = GT_probs / modified_probs  # (batch_size, num_classes)
            # weight = modified_probs / GT_probs  # (batch_size, num_classes)
            # print(weight)

            if self.weighted:
                weight = (probs + self.eps) / (GT_probs + self.eps)  # (batch_size, num_classes)
                # print(probs)
                # print(GT_probs)
                # print(weight)

        
        # if balanced:
        #     balanced_negative_weight = sample_per_class.view(1, -1).repeat(batch_size, 1).to(device)    # (batch_size, num_classes)
        #     balanced_positive_weight = (balanced_negative_weight * mask).sum(dim=1) # (batch_size)
        #     # balanced_weight = balanced_positive_weight.view(-1, 1).repeat(1, num_classes) / balanced_negative_weight    # (batch_size, num_classes)
        #     balanced_weight = balanced_negative_weight / balanced_positive_weight.view(-1, 1).repeat(1, num_classes)    # (batch_size, num_classes)



        # logits = feat.mm(prototypes.T) / self.temp    # (batch_size, num_classes)
        # print(logits)

        logits, _ = torch.matmul(prototypes, feat.T).permute(2, 0, 1).max(dim=2)   # (batch_size, num_classes)
        # print(torch.matmul(prototypes, feat.T).permute(2, 0, 1))
        logits = logits / self.temp
        # print(logits)

        positive_logits = torch.sum(logits*mask, dim=1) # (batch_size)
        # print(positive_logits)
        # negative_logits = torch.sum(torch.exp(logits), dim=1)   # (batch_size)
        # print(negative_logits)

        # if discriminative and balanced:
        #     negative_logits = torch.sum(torch.exp(logits) * weight * balanced_weight, dim=1)   # (batch_size), weighted with hard class mining and rebalancing
        # elif discriminative:
        #     negative_logits = torch.sum(torch.exp(logits) * weight, dim=1)   # (batch_size), weighted with hard class mining
        # elif balanced:
        #     negative_logits = torch.sum(torch.exp(logits) * balanced_weight, dim=1)   # (batch_size), weighted with rebalancing
        # else:
        if self.weighted:
            negative_logits = torch.sum(torch.exp(logits) * weight * dis_mask, dim=1)
        else:
            negative_logits = torch.sum(torch.exp(logits) * dis_mask, dim=1)

        loss = - (positive_logits - torch.log(negative_logits)).mean()

        return loss




def create_loss(temp=0.1, eps=1e-3, weighted=False):
    print('Loading DisPSCLoss.')
    return DisPSCLoss(temp, eps, weighted)

if __name__ == '__main__':
    psc = create_loss(eps=1e-7, weighted=True)
    batch_size = 4
    num_classes = 5
    feat_dim = 10
    feat = torch.randn(batch_size, feat_dim)
    label = torch.tensor([2, 1, 3, 0])
    print(label)
    prototypes = torch.randn(num_classes, 2, feat_dim)
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0],
                          [0.2, 0.4, 0, 0.1, 0.3], 
                          [0.4, 0, 0.3, 0.1, 0.2],
                          [0, 0.1, 0.4, 0.3, 0.2]])
    print(probs)
    loss = psc(feat, label, prototypes, probs, discriminative=True)
    print(loss)

