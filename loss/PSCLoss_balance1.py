import torch.nn as nn
import torch
import torch.nn.functional as F

class PSCLoss(nn.Module):
    def __init__(self, temp=0.1, gamma=1):
        super(PSCLoss, self).__init__()
        self.temp = temp
        self.gamma = gamma
    
    def forward(self, feat, label, prototypes, probs, sample_per_class, discriminative=False, balanced=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        feat = F.normalize(feat, dim=1) # normalize to 1
        batch_size, feat_dim = feat.shape
        num_classes, k, _ = prototypes.shape    # (num_classes, k, feat_dim)
        mask = torch.zeros(batch_size, num_classes).to(device).scatter_(1, label.view(-1, 1), 1) # (batch_size, num_classes)
        # print(mask)
        # print(mask.shape)

        if discriminative:
            eps = 1e-3  # default 1e-3
            GT_probs = (probs*mask).sum(dim=1).view(-1, 1).repeat(1, num_classes) + eps
            # print(GT_probs)
            modified_probs = torch.where(probs > GT_probs, probs, GT_probs)
            # print(modified_probs)
            # weight = GT_probs / modified_probs  # (batch_size, num_classes)
            weight = modified_probs / GT_probs  # (batch_size, num_classes)
            # print(weight)
        
        if balanced:
            balanced_negative_weight = sample_per_class.view(1, -1).repeat(batch_size, 1).to(device)    # (batch_size, num_classes)
            balanced_GT_weight = (balanced_negative_weight * mask).sum(dim=1) # (batch_size)
            # balanced_weight = balanced_positive_weight.view(-1, 1).repeat(1, num_classes) / balanced_negative_weight    # (batch_size, num_classes)
            # balanced_weight = balanced_negative_weight / balanced_positive_weight.view(-1, 1).repeat(1, num_classes)    # (batch_size, num_classes)
            balanced_weight = max(sample_per_class) / balanced_GT_weight    # (batch_size)
            balanced_weight = torch.pow(balanced_weight, self.gamma)
            # print(balanced_weight)



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


        negative_logits = torch.sum(torch.exp(logits), dim=1)

        if balanced:
            loss = - ((positive_logits - torch.log(negative_logits)) * balanced_weight).mean()
        else:
            loss = - (positive_logits - torch.log(negative_logits)).mean()

        return loss




def create_loss(temp=0.1, gamma=1):
    print('Loading PSCLoss.')
    return PSCLoss(temp, gamma)

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
    sample_per_class = torch.tensor(range(1, num_classes+1))
    loss = psc(feat, label, prototypes, probs, sample_per_class, balanced=True)
    print(loss)

