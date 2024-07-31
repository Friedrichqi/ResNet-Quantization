import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import os
import os.path as osp
from tqdm import tqdm
import sys
from qtorch import FixedPoint
from qtorch.quant import Quantizer, quantizer
import matplotlib.pyplot as plt

num_classes = 10
def plot_weight_distribution(layer, block_name, idx_block, idx_model):
    if not osp.exists(f'Weight Distribution CIFAR{num_classes}'):
        os.makedirs(f'Weight Distribution CIFAR{num_classes}')
    
    weights = layer.weight.data.cpu().numpy()
    plt.hist(weights.flatten(), bins=10)
    plt.title(f'Resnet{idx_model}\'s layer{idx_block} {block_name} Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.savefig(f'Weight Distribution CIFAR{num_classes}/Resnet{idx_model}\'s layer{idx_block} {block_name} Weight Distribution.jpg')
    

resnet_models = [resnet18, resnet34, resnet50, resnet101, resnet152]
idx_list = [18, 34, 50, 101, 152]
for idx, resnet_model in enumerate(resnet_models):
    model = resnet_model()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(f'/data/home/qyjh/Quantization_evaluation/Resnet_Test/model/Resnet{idx_list[idx]}_CIFAR{num_classes}.pt')
    model.load_state_dict(state_dict)
    
    block_list = [model.layer1, model.layer2, model.layer3, model.layer4]
    for idx_block, block in enumerate(block_list):
        for name, module in block.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
                plot_weight_distribution(module, name, idx_block+1, idx_list
                                         [idx])
                
                