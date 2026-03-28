import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import torch.nn as nn
from dataset import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import clip
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description='Optimize Selection Parameters')
parser.add_argument('--dataset', type=str, default='CIFAR100', help='Name of the dataset')
args = parser.parse_args()
DATASET = args.dataset

DATASET_NUM = {
    'CIFAR10': 50000,
    'CIFAR100': 50000,
}
DATASET_NCLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
}
def pdist_torch(features):
    with torch.no_grad():
        features_sq = torch.sum(features ** 2, dim=1, keepdim=True)
        dist_sq = features_sq + features_sq.t() - 2 * torch.mm(features, features.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        dist = torch.sqrt(dist_sq + 1e-6)
        del dist_sq
        del features_sq
        return dist
    
def extract_img_feature_space(DATASET):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.float()
    transform = transforms.Compose([
        preprocess
    ])
    if DATASET == 'CIFAR100':
        train_dataset = CIFAR100(root='./data/', train=True, download=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=16)
    elif DATASET == 'CIFAR10':
        train_dataset = CIFAR10(root='./data/', train=True, download=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=16)
 
    input_dim = model.text_projection.shape[1]
    with torch.no_grad():
        feature_map_list = torch.zeros(len(train_loader.dataset),input_dim).cuda()
        if DATASET == 'CIFAR10':
            adpater_img = nn.Linear(input_dim, input_dim).cuda()
            adpater_img.load_state_dict(torch.load(f'./adapter_ckpt/{DATASET}/adapater_img.pth'))
        elif DATASET == 'CIFAR100':
            adpater_img = nn.Linear(input_dim, input_dim).cuda()
            adpater_img.load_state_dict(torch.load(f'./adapter_ckpt/{DATASET}/adapater_img.pth'))
        adpater_img.eval()
        for i, (index, images, target) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            batch_image_features = model.encode_image(images)
            feature_map = adpater_img(batch_image_features)
            feature_map_list[index] = feature_map
    return feature_map_list

def optimize_mask(similarity_scores, dis_loss, sr):
    lambda_ = 0.1 # balance factor
    beta_ = 2 # increase to speed up the selection process
    learning_rate = 0.001  # learning rate
    num_epochs = 100000 # iteration steps
    # Initialize weights
    n = len(similarity_scores)
    w = nn.Parameter(0.01 * torch.ones(n, requires_grad=True).cuda())
    # Optimizer
    optimizer = optim.SGD([w], lr=learning_rate, momentum=0.9)
    
    k = int(n*sr)
    scale_factor = 100.
    for epoch in range(num_epochs):
        loss1 = - torch.mean(torch.sigmoid(scale_factor * w) * (similarity_scores / similarity_scores.mean()))
        loss2 = - torch.mean(torch.sigmoid(scale_factor * w) * (dis_loss / dis_loss.mean())) * lambda_
        x = torch.sigmoid(scale_factor * w)
        loss3 = torch.sqrt(((((x > 0.5).float() - x.detach() + x).sum() - k)/n) ** 2) * beta_
        loss = loss1 + loss2 + loss3
        if epoch % 50 == 0:
            print(sr, ' Epoch:',epoch,'Loss:',loss.item(),'Loss1:',loss1.item(),'Loss2:',loss2.item(),'Loss3:',loss3.item(),'sr:',len(torch.where(torch.sigmoid(scale_factor * w)>0.5)[0])/n)
        if loss3.item() < 0.001: #0.02
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scores = torch.sigmoid(scale_factor * w).cpu().detach()
    zero = torch.zeros_like(scores)
    one = torch.ones_like(scores)
    reserve = torch.where(scores > 0.5,one,zero).numpy()
    file_name = f'./selection_res_sr_{int(sr*100)}.npy'
    # np.save(file_name,reserve)

def calculate_scores():
    if DATASET == 'CIFAR100':
        similarity_scores = np.load(f'./Pruning_Scores/{DATASET}/scores.npy')
    elif DATASET == 'CIFAR10':
        similarity_scores = np.load(f'./Pruning_Scores/{DATASET}/scores.npy')
    similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())
    class_num = DATASET_NCLASSES[DATASET]
    total = DATASET_NUM[DATASET]
    sample_num_per_class = total // class_num
    dis_loss = torch.zeros(DATASET_NUM[DATASET]).cuda()
    feature_map_list = torch.load(f'../checkpoint/{DATASET}/feature_map_list.pt')
    if DATASET == 'CIFAR100':
        K = 50
    elif DATASET == 'CIFAR10':
        K = 100
    from sklearn.neighbors import NearestNeighbors 
    for i in range(DATASET_NCLASSES[DATASET]):
        start = i * sample_num_per_class
        end = (i + 1) * sample_num_per_class
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(feature_map_list[start:end].cpu().numpy())
        distances, _ = nbrs.kneighbors(feature_map_list[start:end].cpu().numpy())
        nearest_neighbor_distances = distances[:, 1:]
        dis_loss[start:end] = torch.mean(torch.tensor(nearest_neighbor_distances),dim=1).cuda()
    dis_loss = (dis_loss - dis_loss.min()) / (dis_loss.max() - dis_loss.min())
    return similarity_scores, dis_loss


if __name__ == '__main__':
    similarity_scores, dis_loss = calculate_scores()
    sr_list = [0.9]
    for sr in sr_list:
        optimize_mask(torch.tensor(similarity_scores).cuda(), dis_loss, sr)
 


