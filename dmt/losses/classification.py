""" Module classification.py (By: Charley Zhang, Nov 2020)
Collection of multi-class, multi-label, and reconstruction losses for 
general image classification tasks.

Resources:
https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
"""

import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['HardCrossEntropy', 'SoftCrossEntropy', 'SoftmaxMSE']



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        print(f" (FocalLoss) Initiated with alpha {alpha} & gamma {gamma}.")

    def forward(self, output, target):
        '''Focal loss.
        Args:
          output: (tensor) sized [N,D].
          target: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = self.alpha
        gamma = self.gamma
        x = output
        
        to_one_hot = torch.eye(output.shape[1])
        t = to_one_hot[target.data.cpu()].to(output.device)

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = (w * (1-pt).pow(gamma)).detach()
        loss = F.binary_cross_entropy_with_logits(x, t, weight=w, reduction='sum')
        return loss

    
class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, lossname='focal', beta=0.999, gamma=1):
        """ https://github.com/vandit15/Class-balanced-loss-pytorch/
        Parameters
          samples_per_class: A python list of size [no_of_classes].
          lossname: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        """
        assert lossname in ('sigmoid', 'focal', 'softmax')
        
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = samples_per_class
        self.lossname = lossname
        self.beta = beta
        self.gamma = gamma
        
        print(f" (ClassBalancedLoss) Initiated with class_counts={samples_per_class}\n"
              f"\t loss={lossname}, beta={beta}, gamma={gamma}.")
    
    def forward(self, logits, labels):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Parameters
            logits: A float tensor of size [batch, no_of_classes].
            labels: A int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """
        loss_type = self.lossname
        beta = self.beta
        gamma =self.gamma
        
        no_of_classes = logits.shape[1]
        samples_per_cls = self.samples_per_class
        
        # Original Implementation
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        labels_one_hot = F.one_hot(labels, no_of_classes).float().to(logits.device)

        weights = torch.tensor(weights).float().to(logits.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)

        if loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input = logits,target = labels_one_hot, 
                weights = weights
            )
        elif loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(
                input = pred, target = labels_one_hot, 
                weight = weights
            )
        return cb_loss
    
    def focal_loss(self, labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(
            input = logits, target = labels,
            reduction = "none"
        )

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss



class HardCrossEntropy(nn.Module):
	""" Default nn.CrossEntropy with hard integer labels.
	Expects input & target to be BxC (batchsize x num_classes).
	"""

	def __init__(self, weights=None, reduction='mean', ignore_index=-100):
		assert reduction in ['none', 'mean', 'sum'], f"{reduction} not valid."
		
		super(HardCrossEntropy, self).__init__()
		self.reduction = reduction
		self.weights = torch.tensor(weights).float() if weights is not None else None
		self.ce = nn.CrossEntropyLoss(weight=self.weights, reduction=reduction,
			ignore_index=ignore_index) 
		print(f"\t (HardCE) Initialized w/{reduction} reduction & weights: {weights}")

	def forward(self, output, target):
		"""
		Parameters:
			output (BxC float tensor) - logits or probs for each batch example
			target (B long tensor) - class index 
		"""
		if self.weights is not None:
			assert self.weights.shape[0] == output.shape[1]
		assert output.shape[0] == target.shape[0], (output.shape, target.shape)
		
		# if logits:
		# 	output = output.softmax(-1)
		# assert output.sum().abs().item() - output.shape[0] < 10**-5, \
		# 	(output.sum(), output.shape[0])

		loss = self.ce(output, target.long())
		return loss


class SoftCrossEntropy(nn.Module):
	""" Real statistical CrossEntropy that deals with 2 batched probability
	distributions. 
	"""

	def __init__(self, reduction='mean', weights=None):
		assert reduction in ['none', 'mean', 'sum'], f"{reduction} not valid."
		
		super(SoftCrossEntropy, self).__init__()
		self.reduction = reduction
		self.weights = torch.tensor(weights) if weights is not None else None
		print(f"\t (SoftCE) Initialized w/{reduction} reduction & weights: {weights}")

	def forward(self, output, target):
		"""
		Parameters:
			output (BxC float tensor) - model output logits
			target (BxC float tensor) - target distribution
		"""
		N, C = output.shape
		if self.weights is not None:
			assert self.weights.shape[0] == output.shape[1] == target.shape[1]

		assert output.sum().abs().item() - N < 10**-5
		assert target.sum().abs().item() - N < 10**-5

		logprobs = F.log_softmax(output, dim=1)
		cum_loss = -(target * logprobs)

		if self.reduction == 'none':
			return cum_loss
		elif self.reduction == 'sum':
			return cum_loss.sum()
		else:
			return cum_loss.sum() / N


class MSE(nn.Module):
	
	def __init__(self, reduction='mean'):
		assert reduction in ['none', 'mean', 'sum'], f"{reduction} not valid."

		super(MSE, self).__init__()
		self.mse = nn.MSELoss(reduction=reduction)
		print(f"\t (MSE) Initialized w/{reduction} reduction.")

	def forward(self, output, target):
		loss = self.mse(output, target)
		return loss


class SoftmaxMSE(nn.Module):
	
	def __init__(self, weights=None, reduction='mean'):
		assert reduction in ['none', 'mean', 'sum'], f"{reduction} not valid."
		
		super(SoftmaxMSE, self).__init__()
		self.reduction = reduction
		self.weights = torch.tensor(weights) if weights is not None else None
		print(f"\t (SoftmaxMSE) Initialized w/{reduction} reduction & weights: {weights}")

	def forward(self, output, target):
		"""
		Parameters:
			output (BxC float tensor) - network output logits
			target (BxC float tensor) - target logits
		"""
		assert output.shape == target.shape, (output.shape, target.shape)
		N, C = target.shape
		output_softmax = F.softmax(output, dim=1)
		target_softmax = F.softmax(target, dim=1)

		loss = (output_softmax - target_softmax) ** 2
		if self.weights is not None:
			loss = loss * self.weights.to(loss.device)
		
		if self.reduction == 'none':
			return loss
		elif self.reduction == 'sum':
			return loss.sum()
		else:
			return loss.sum() / (target.shape[0] * target.shape[1])
		
		# Experimental Code.. (not run)
		mse = F.mse_loss(output_softmax, target_softmax, reduction='none').sum(1) / C
		ent_mask = (- output_softmax * torch.log(output_softmax)).sum(1) > 0.1
		out = mse * ent_mask.float()

		return out.sum() / ent_mask.count_nonzero()
		



def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss

