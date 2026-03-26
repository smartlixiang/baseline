import os
import time
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--gpu', default='3', type=str, help='GPU id to use.')
parser.add_argument('--dataset', default='CIFAR100', type=str)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
import clip
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import dataset
from torch.utils.data import DataLoader
import numpy as np
from utils import *
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

def text_(dataset):
    class_names = obtain_classnames(dataset)
    text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features
 
def load_pretrained_adapter_for_pruning():
    adpater_img = adpater_text = nn.Linear(512, 512).cuda()
    adpater_img.load_state_dict(torch.load(f'./adapter_ckpt/{args.dataset}/adapater_img.pth'))
    adpater_text.load_state_dict(torch.load(f'./adapter_ckpt/{args.dataset}/adapater_text.pth'))
    adpater_img.eval()
    adpater_text.eval()
    transform = transforms.Compose([
        preprocess
    ])
    train_dataset = getattr(dataset, args.dataset)(root='./data/', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=8)
    with torch.no_grad():   
        ft_text_features = adpater_text(text_(args.dataset))
    scores = torch.ones(len(train_dataset)) * -1
    for i, (index, images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)
        # Extract features from CLIP model
        with torch.no_grad(): 
            batch_image_features = model.encode_image(images)
            # Embedding features by adapter
            image_outputs = adpater_img(batch_image_features)
            # batchsize * 512
            batch_text_features = ft_text_features[target] 
            # Calculate Cosine Similarity
            matchness = F.cosine_similarity(image_outputs.cpu(), batch_text_features.cpu())
            scores[index] = matchness.cpu()
    # np.save(f'./Pruning_Scores/{args.dataset}/scores.npy',scores.numpy())
 
    
if __name__ == '__main__':
    load_pretrained_adapter_for_pruning()