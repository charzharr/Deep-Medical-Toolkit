
import sys
import torch, torch.nn as nn


class CrossEntropy(nn.Module):

    def __init__(self, weights=None):
        super(CrossEntropy, self).__init__()
        self.weights = torch.tensor(weights).to(torch.float32) if weights \
            else None
        self.CE = nn.CrossEntropyLoss(weight=self.weights)
        print(f"\tCE loss initialized (weights:{weights}).")
    
    def forward(self, preds, targs):
        if targs.ndim == 4:
            targs = targs.squeeze()
        assert targs.ndim == 3, f"Invalid target dims: {targs.shape}"
        
        loss = self.CE(preds, targs.long())
        return loss


class FocalLoss(nn.Module):
    """ L(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
          p_t: prediction prob for target class, t
    Paper: Focal Loss for Dense Object Detection
        https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, preds, targs, eps=10**-6):
        B = pred.shape[0]
        preds = self.softmax(preds)       
        preds = inputs.view(B, -1)
        targets = targets.view(B, -1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP)**self.gamma * BCE
                       
        return focal_loss


class SoftDiceLoss(nn.Module):

    def __init__(self, square_denom=False):
        super(SoftDiceLoss, self).__init__()
        self.square_denom = square_denom
        self.softmax = nn.Softmax(dim=1)
        print('\tSoftDice initialized.')

    def forward(self, pred, targ, eps=1):
        """
        Parameters
            pred: BxCxHxW logits as a float tensor on the same device
            targ: BxCxHxW binary labels as a float tensor
        """
        assert pred.shape == targ.shape, f"Pred: {pred.shape}, Targ: {targ.shape}"
        assert pred.dtype == torch.float32
        
        B, C = pred.shape[0], pred.shape[1]
        pred_probs = pred.sigmoid() if C == 1 else self.softmax(pred)
        preds = pred_probs.view(B, -1)
        targs = targ.view(B, -1)
        
        intersect = preds * targs
        if self.square_denom:
            dice = (2. * intersect.sum(1) + eps) / (preds.square().sum(1) + \
                    targs.square().sum(1) + eps)
        else:
            dice = (2. * intersect.sum(1) + eps) / (preds.sum(1) + \
                    targs.sum(1) + eps)
        
        return 1 - dice.mean()


class SoftJaccardLoss(nn.Module):

    def __init__(self, log_loss=False):
        """
        log_loss = -ln(IOU) as presented in this paper:
            https://arxiv.org/pdf/1608.01471.pdf
        """
        super(SoftJaccardLoss, self).__init__()
        self.log_loss = log_loss
        self.softmax = nn.Softmax(dim=1)
        print(f'\tSoftJaccard initialized (log_jaccard: {log_loss}).')

    def forward(self, pred, targ, eps=10**-6):
        """
        Parameters
            pred: BxCxHxW logits as a float tensor on the same device
            targ: BxCxHxW binary labels as a float tensor
        """
        assert pred.shape == targ.shape, f"Pred: {pred.shape}, Targ: {targ.shape}"
        assert pred.dtype == torch.float32
        
        B, C = pred.shape[0], pred.shape[1]
        pred_probs = pred.sigmoid() if C == 1 else self.softmax(pred)
        preds = pred_probs.view(B, -1)
        targs = targ.view(B, -1)
        
        intersect = torch.sum(preds * targs, 1)
        union = torch.sum(preds + targs, 1) - intersect  # B x C tensor
        iou = torch.mean((intersect + eps) / (union + eps))
        return 1 - iou if not self.log_loss else -torch.log(iou)


class FocalTverskyLoss(nn.Module):
    """ Generalized loss that is a superset of dice, jaccard, and focal loss."""

    def __init__(self, alpha=0.5, beta=0.5, gamma=1):
        """ L = (1 - TI)^gamma, TI = TP / (TP + alpha*FN + beta*FP) 
        Quick Notes
            - alpha = beta = 0.5 is dice loss
            - alpha = beta = 1. is jaccard loss
            - gamma > 1 gives concave loss, gamma < 1 gives convex
            - good default for FN-focus: alpha=0.7, beta=0.3, gamma=3/4
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        print(f"\tTversky loss initialized (α:{alpha}, β:{beta}, γ:{gamma}).")

    def forward(self, pred, targ, eps=1):
        assert pred.shape == targ.shape, f"Pred{pred.shape}, Targ{targ.shape}"
        assert pred.ndim == 4, f"4D tensor required. Got: {pred.shape}."
        
        B, C = pred.shape[0], pred.shape[1]
        if C == 1:
            pred_probs = pred.sigmoid()
        else:
            smax = torch.nn.Softmax(dim=1)
            pred_probs = smax(pred)
        preds = pred_probs.view(B, -1)
        targets = targ.view(B, -1)
        
        TP = (preds * targets).sum(1)
        FN = (targets * (1 - preds)).sum(1)
        FP = ((1 - targets) * preds).sum(1)
        
        tversky = (TP + eps) / (TP + self.alpha * FN + self.beta * FP + eps)
        return (1 - torch.mean(tversky)) ** self.gamma
