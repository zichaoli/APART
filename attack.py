import os
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

class LinfPGDAttack(object):
    """
        Attack parameter initializa1on. The attack performs k steps of size
        alpha, while always staying within epsilon from the initial point.
            IFGSM(Iterative Fast Gradient Sign Method) is essentially
            PGD(Projected Gradient Descent)
    """

    def __init__(self, epsilon=8/255, k=10, alpha=2/255, random_start=True):
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start

    def __call__(self, model, x, y, k=None):
        self.model = model
        if k is not None:
            self.k = k
        if self.random_start:
            x_adv = x + x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            x_adv = x.clone()
        training = self.model.training
        if training:
            self.model.eval()
        for i in range(self.k):
            self.model.zero_grad()
            x_adv.requires_grad_()
            loss_f = nn.CrossEntropyLoss()
            pred = loss_f(self.model(x_adv), y)
            pred.backward()
            grad = x_adv.grad
            x_adv = x_adv.detach() + (self.alpha) * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)

            x_adv.clamp_(0, 1)
        if training:
            self.model.train()
        return x_adv