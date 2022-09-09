import torch.nn as nn
import torch
import torch.nn.functional as F

class PSCLoss(nn.Module):
    def __init__(self, temp=0.1):
        super(PSCLoss, self).__init__()
        self.temp = temp
    
    def forward(self, feat, label, prototypes, probs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        feat = F.normalize(feat, dim=1) # normalize to 1
        batch_size, feat_dim = feat.shape
        num_classes, k, _ = prototypes.shape    # (num_classes, k, feat_dim)
        mask = torch.zeros(batch_size, num_classes).to(device).scatter_(1, label.view(-1, 1), 1) # (batch_size, num_classes)
        # print(mask)
        # print(mask.shape)

        eps = 1e-3
        GT_probs = (probs*mask).sum(dim=1).view(-1, 1).repeat(1, num_classes) + eps
        # print(GT_probs)
        modified_probs = torch.where(probs > GT_probs, probs, GT_probs)
        # print(modified_probs)
        weight = GT_probs / modified_probs  # (batch_size, num_classes)
        # print(weight)


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

        negative_logits = torch.sum(torch.exp(logits) * weight, dim=1)   # (batch_size), weighted with hard class mining

        loss = - (positive_logits - torch.log(negative_logits)).mean()

        return loss




def create_loss(temp=0.1):
    print('Loading PSCLoss.')
    return PSCLoss(temp)

if __name__ == '__main__':
    psc = create_loss()
    batch_size = 4
    num_classes = 5
    feat_dim = 10
    feat = torch.randn(batch_size, feat_dim)
    label = torch.tensor([2, 1, 3, 0])
    prototypes = torch.randn(num_classes, 2, feat_dim)
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0],
                          [0.2, 0.4, 0, 0.1, 0.3], 
                          [0.4, 0, 0.3, 0.1, 0.2],
                          [0, 0.1, 0.4, 0.3, 0.2]])
    loss = psc(feat, label, prototypes, probs)
    print(loss)

