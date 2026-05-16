from .ResNet import *

def get_model(model, num_classes):
    if model == 'resnet18':
        return ResNet18(num_classes)