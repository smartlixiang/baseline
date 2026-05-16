import os
import time
import random
import logging
import argparse
import numpy as np

from A2C import A2C
from tqdm import tqdm
from Network import *
from Dataset import *
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--log_interval", type=int, default=50, help="log training status")
parser.add_argument("--weight-decay", "--wd", default=5e-4, type=float, metavar="W", help="weight decay (default: 5e-4)")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--cutout_length", type=int, default=16)
parser.add_argument("--root", type=str, default="/data1/ysf/CIFAR100/data")
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--num_worker", type=int, default=8, choices=[2, 4, 8, 16, 32])
parser.add_argument("--compression_rate", type=float, default=0.8)
parser.add_argument("--epoches", type=int, default=200)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--action_dim", type=int, default=1)
parser.add_argument("--state_dim", type=int, default=512)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def setup_seed(seed):
    if seed == None:
        seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(args.seed)

now = time.strftime("%m%d%H%M")
log_name = './Log/%s/resnet18_%d_seed%d_%s.log' % (args.dataset, args.compression_rate * 100, args.seed, now)

logging.basicConfig(
    filename=log_name,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

DELTA = {"CIFAR10": 5, "CIFAR100": 5, "Tiny-Imagenet": 10}

MASK_PRUNER = torch.ones(DATASET_NUM[args.dataset])
FEATURE_MAP = torch.zeros(DATASET_NUM[args.dataset], args.state_dim).to(device)
target_cr = args.compression_rate

model = get_model(args.model, num_classes=DATASET_NCLASSES[args.dataset])
model = model.to(device)

if args.resume:
    path = './checkpoint/{}/{}_best.pth'.format(args.dataset, args.model)
    state_dict = torch.load(path)['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

agent = A2C(args).to(device)

trainset, testset = init_dataset(args.root, args.dataset)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoches)

acc_list = []
loss = nn.CrossEntropyLoss(reduction="none")

start_epoch = 0
args.total = len(trainloader.dataset)

def save_mask(filename, epoch, acc):
    zero = torch.zeros_like(MASK_PRUNER)
    one = torch.ones_like(MASK_PRUNER)
    mask = torch.where(MASK_PRUNER > 0.5, one, zero)
    state = {
        "epoch": epoch,
        "acc": acc,
        "mask": mask,
    }
    torch.save(state, "./Saved_Mask/{}/{}.pt".format(args.dataset, filename))


def pre_train_epoch(net):
    global optimizer
    global loss
    net.train()
    for data in trainloader:
        idx, inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, feature = net(inputs)
        FEATURE_MAP[idx] = feature.detach()
        crossloss = loss(outputs, labels).mean()
        optimizer.zero_grad()
        crossloss.backward()
        optimizer.step()


def compute_distance_matrix(index_map, feature_map):
    dis_loss = torch.zeros(DATASET_NUM[args.dataset])
    for _, idx in index_map.items():
        feature = feature_map[idx]
        feature_sq = torch.sum(feature**2, dim=1, keepdim=True)
        dist_sq = feature_sq + feature_sq.t() - 2 * feature @ feature.t()
        dist_sq = torch.clamp(dist_sq, min=0.0)
        dist = torch.sqrt(dist_sq)
        dis_loss[idx] = dist.mean(axis=0).cpu()
    return dis_loss


def train(net, agent, epoch):
    global optimizer
    global loss
    global best_r2
    net.train()
    training_loss = 0.0
    total = len(trainloader.dataset)
    correct = 0
    total_reward = 0
    total_r1 = 0
    total_r2 = 0
    # 得到每个样本对应的类内样本对distance
    dis_loss = compute_distance_matrix(trainset.class2index, FEATURE_MAP)
    # 对距离进行归一化
    for i, data in enumerate(trainloader):

        mask_index, inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)
        outputs, feature = net(inputs)

        state = feature.detach().squeeze()
        action = agent.action(state)
        mask = action.detach().squeeze()

        MASK_PRUNER[mask_index] = mask.cpu()
        FEATURE_MAP[mask_index] = state

        crossloss = loss(outputs, labels)
        selected_loss = (crossloss * mask).mean()
        optimizer.zero_grad()
        selected_loss.backward()
        optimizer.step()

        # batch内压缩率
        # current_cr = torch.where(MASK_PRUNER[mask_index] > 0.5)[0].shape[0] / MASK_PRUNER[mask_index].shape[0]
        # 全局压缩率
        current_cr = torch.where(MASK_PRUNER > 0.5)[0].shape[0] / MASK_PRUNER.shape[0]
        gap = current_cr - target_cr
        max_gamma = 1 - target_cr if gap >= 0 else target_cr
        gamma = abs(gap) / max_gamma  # 将gap映射到[0,1]之间，越大说明差距越大
        reward_1 = DELTA[args.dataset] * (1 - gamma)
        # batch内距离
        # reward_2 = (dis_loss[mask_index] * MASK_PRUNER[mask_index]).mean().item()
        # 全局距离
        reward_2 = (dis_loss * MASK_PRUNER).mean().item()
        reward = reward_1 + reward_2
        agent.update(state, action, reward)

        total_reward += reward
        total_r1 += reward_1
        total_r2 += reward_2
        training_loss += selected_loss.mean().item()
        predicted = outputs.max(1).indices
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            loss_mean = training_loss / (i + 1)
            trained_total = (i + 1) * len(labels)
            current_compression_rate = torch.where(MASK_PRUNER > 0.5)[0].shape[0] / MASK_PRUNER.shape[0]
            progress = 100.0 * trained_total / total
            acc = correct * 100.0 / trained_total
            logging.info(
                "Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.3f} ACC: {:.2f} CR: {:.2f} Avg R1: {:.2f} Avg R2: {:.2f} Avg Reward: {:.6f}".format(
                    epoch,
                    trained_total,
                    total,
                    progress,
                    loss_mean,
                    acc,
                    current_compression_rate,
                    total_r1 / (i + 1),
                    total_r2 / (i + 1),
                    total_reward / (i + 1),
                )
            )
    current_compression_rate = torch.where(MASK_PRUNER > 0.5)[0].shape[0] / MASK_PRUNER.shape[0]
    logging.info("EPOCH:{}, ======================CR:{}====================".format(epoch, current_compression_rate))
    logging.info(np.corrcoef(dis_loss.numpy(), MASK_PRUNER.numpy())[0, 1])
    return current_compression_rate, total_r2


def test(net, epoch):
    global best_acc
    global best_epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            predicted = outputs.max(1).indices
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct * 100.0 / total
    logging.info("EPOCH:{}, ======================ACC:{}====================".format(epoch, acc))
    acc_list.append(acc)
    if acc >= best_acc:
        best_acc = acc
        best_epoch = epoch
    logging.info("BEST EPOCH:{},BEST ACC:{}".format(best_epoch, best_acc))
    return acc

THRESHOLD = 0.01
best_r2 = 0
best_acc = 0
best_saved_acc = 0.0
best_epoch = 0

if __name__ == "__main__":
    average_time = 0
    warmup = 1

    for epoch in tqdm(range(start_epoch, args.epoches)):
        if epoch < warmup:
            pre_train_epoch(model)
        current_compression_rate, total_r2 = train(model, agent, epoch)
        test_acc = test(model, epoch)
        scheduler.step()
        agent.lr_decay(epoch)

        if np.abs(current_compression_rate - args.compression_rate) <= THRESHOLD:
            logging.info("Saving Masks:", current_compression_rate, args.compression_rate)
            # 根据不同的状态保存多个mask做消融
            if test_acc >= best_saved_acc:
                best_saved_acc = test_acc
                best_epoch = epoch
                save_mask(
                    "pretrained-rescale_mask_best-acc_{}_{}_{}".format(
                        str(int(args.compression_rate * 100)),
                        now,
                        args.model,
                    ),
                    epoch,
                    test_acc,
                )
            if total_r2 >= best_r2:
                best_r2 = total_r2
                save_mask(
                    "pretrained-rescale_mask_best-r2_{}_{}_{}".format(
                        str(int(args.compression_rate * 100)),
                        now,
                        args.model,
                    ),
                    epoch,
                    test_acc,
                )
        # agent.save_policy(args.compression_rate, './')