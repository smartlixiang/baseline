import os
import time
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import sys

from A2C import A2C
from tqdm import tqdm
from Network import *
from Dataset import *
from collections import OrderedDict

REPO_ROOT = Path(__file__).resolve().parents[1]

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
parser.add_argument("--root", type=str, default=str(REPO_ROOT / "data"))
parser.add_argument("--output_root", type=str, default=str(REPO_ROOT / "mask"))
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--num_worker", type=int, default=8, choices=[2, 4, 8, 16, 32])
parser.add_argument("--compression_rate", type=float, default=None)
parser.add_argument("--keep_ratio", type=int, default=None, choices=[20, 30, 40, 50, 60, 70, 80, 90])
parser.add_argument("--epoches", type=int, default=200)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--action_dim", type=int, default=1)
parser.add_argument("--state_dim", type=int, default=512)
raw_argv = sys.argv[1:]
user_set_epoches = any(
    x == '--epoches' or x.startswith('--epoches=')
    for x in raw_argv
)
user_set_weight_decay = any(
    x in ['--weight-decay', '--wd'] or x.startswith('--weight-decay=') or x.startswith('--wd=')
    for x in raw_argv
)

args = parser.parse_args()

args.dataset = normalize_dataset_name(args.dataset)
args.dataset_out = dataset_output_name(args.dataset)

# Paper setting for Tiny-ImageNet:
# total epochs = 90, weight decay = 1e-4, lr decays after epochs 30 and 60.
if args.dataset == 'Tiny-Imagenet':
    if not user_set_epoches:
        args.epoches = 90
    if not user_set_weight_decay:
        args.weight_decay = 1e-4

if args.keep_ratio is not None:
    args.compression_rate = args.keep_ratio / 100.0
elif args.compression_rate is not None:
    args.keep_ratio = int(round(args.compression_rate * 100))
    if args.keep_ratio not in [20, 30, 40, 50, 60, 70, 80, 90]:
        raise ValueError("compression_rate must correspond to keep_ratio in [20,30,40,50,60,70,80,90].")
else:
    args.keep_ratio = 80
    args.compression_rate = 0.8

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
    if seed is None:
        seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(args.seed)

log_dir = Path(__file__).resolve().parent / "Log" / args.dataset
log_dir.mkdir(parents=True, exist_ok=True)
now = time.strftime("%m%d%H%M")
log_name = log_dir / '{}_{}_seed{}_{}.log'.format(args.model, args.keep_ratio, args.seed, now)

logging.basicConfig(
    filename=str(log_name),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("Args: %s", vars(args))

DELTA = {"CIFAR10": 5, "CIFAR100": 5, "Tiny-Imagenet": 10}

MASK_PRUNER = torch.ones(DATASET_NUM[args.dataset])
FEATURE_MAP = torch.zeros(DATASET_NUM[args.dataset], args.state_dim).to(device)
target_cr = args.compression_rate
STANDARD_MASK_SAVED = False

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

if len(trainset) != DATASET_NUM[args.dataset]:
    raise RuntimeError(
        'Dataset size mismatch for {}: expected {}, got {}'.format(
            args.dataset, DATASET_NUM[args.dataset], len(trainset)
        )
    )

if len(testset) == 0:
    raise RuntimeError(
        'Validation/test set is empty for {}. Please check dataset root: {}'.format(
            args.dataset, args.root
        )
    )

logging.info("Loaded dataset %s: train=%d, test=%d", args.dataset, len(trainset), len(testset))

trainloader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_worker,
    pin_memory=torch.cuda.is_available(),
)

testloader = DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_worker,
    pin_memory=torch.cuda.is_available(),
)

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
if args.dataset == 'Tiny-Imagenet':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60],
        gamma=0.1,
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoches,
    )

acc_list = []
loss = nn.CrossEntropyLoss(reduction="none")

start_epoch = 0
args.total = len(trainloader.dataset)


def _standard_mask_path():
    output_dir = Path(args.output_root) / args.dataset_out / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / 'mask_{}.npz'.format(args.keep_ratio)


def save_standard_mask(epoch, acc, reason):
    """Save exactly keep_ratio% samples as a 0-1 npz mask.

    The internal RL score remains MASK_PRUNER. For the public result file, we use
    top-k so that mask.sum() is exactly round(N * keep_ratio / 100).
    """
    global STANDARD_MASK_SAVED
    scores = MASK_PRUNER.detach().cpu()
    num_samples = scores.numel()
    k = int(round(num_samples * args.keep_ratio / 100.0))
    k = max(0, min(k, num_samples))
    mask = torch.zeros(num_samples, dtype=torch.uint8)
    if k > 0:
        selected = torch.topk(scores, k=k, largest=True).indices
        mask[selected] = 1

    save_path = _standard_mask_path()
    np.savez(save_path, mask=mask.numpy().astype(np.uint8))
    STANDARD_MASK_SAVED = True
    logging.info(
        "Saved standard mask to %s, epoch=%s, acc=%.4f, reason=%s, selected=%d/%d",
        str(save_path), epoch, acc, reason, int(mask.sum().item()), num_samples,
    )


def save_mask(filename, epoch, acc):
    # Keep the old function name used by the original code path, but save the
    # public result in the unified baseline format instead of Saved_Mask/*.pt.
    save_standard_mask(epoch=epoch, acc=acc, reason=filename)


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
    # 得到每个样本对应的类内样本对 distance
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
            logging.info("Saving Masks: current_cr=%.4f, target_cr=%.4f", current_compression_rate, args.compression_rate)
            # 根据不同的状态保存多个mask做消融。对外统一保存到 mask/[dataset]/[seed]/mask_[keep_ratio].npz。
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

    if not STANDARD_MASK_SAVED:
        save_standard_mask(epoch=args.epoches - 1, acc=acc_list[-1] if acc_list else 0.0, reason="final_fallback")
