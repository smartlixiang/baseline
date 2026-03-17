import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, labels_onehot):
    targets = labels_onehot.argmax(dim=-1)
    return F.cross_entropy(logits, targets)


def correct(logits, labels_onehot):
    return logits.argmax(dim=-1) == labels_onehot.argmax(dim=-1)


def accuracy(logits, labels_onehot):
    return correct(logits, labels_onehot).float().mean()
