import os
import time
import random
import logging
import argparse
import numpy as np

from Model import *
from Dataset import *
from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--log_interval", type=int, default=50, help="log training status")
parser.add_argument("--weight-decay", "--wd", default=5e-4, type=float, metavar="W", help="weight decay (default: 5e-4)")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--cutout_length", type=int, default=16)
parser.add_argument("--root", type=str, default="/data1/ysf/CIFAR100/data")
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_argument("--epoches", type=int, default=200)
parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import torch
import torch.nn as nn
from torch import optim

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
log_name = './Log/%s/%s_seed%d.log' % (args.dataset, args.file[:-3], args.seed)

logging.basicConfig(
    filename=log_name,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

model = ResNet50(num_classes=DATASET_NCLASSES[args.dataset])
model = model.to(device)

MASK = torch.load('/data1/ysf/RePruner/Saved_Mask/{}/{}'.format(args.dataset, args.file))['mask'].cpu()
mask = torch.where(MASK == 1)[0]
trainloader, testloader = init_dataset(args.root, args.dataset, args.batch_size, mask)

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoches)
scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier = 1,
    total_epoch = 5,
    after_scheduler = scheduler
)

criterion = nn.CrossEntropyLoss()

def train(net, epoch):
    net.train()
    training_loss = 0.0
    total = len(trainloader.dataset)
    correct = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.mean().item()
        predicted = outputs.max(1).indices
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % args.log_interval == 0:
            loss_mean = training_loss / (i + 1)
            trained_total = (i + 1) * len(labels)
            progress = 100.0 * trained_total / total
            acc = correct * 100.0 / trained_total
            logging.info(
                "Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.3f} ACC: {:.2f}".format(
                    epoch,
                    trained_total,
                    total,
                    progress,
                    loss_mean,
                    acc,
                )
            )


def test(net, epoch):
    global best_acc
    global best_epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            predicted = outputs.max(1).indices
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct * 100.0 / total
    logging.info("EPOCH:{}, ======================ACC:{}====================".format(epoch, acc))
    acc_list.append(acc)
    if acc >= best_acc:
        best_acc = acc
        best_epoch = epoch
    logging.info("BEST EPOCH:{}, BEST ACC:{}".format(best_epoch, best_acc))
    return acc


best_acc = 0.0
best_epoch = 0
acc_list = []
start_epoch = 0

if __name__ == "__main__":
    for epoch in range(start_epoch, args.epoches):
        train(model, epoch)
        test_acc = test(model, epoch)
        scheduler.step()