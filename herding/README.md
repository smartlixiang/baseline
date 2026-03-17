# Herding Baseline (CIFAR-10 / CIFAR-100 / Tiny-ImageNet)

本目录实现了一个简洁的 **class-wise feature herding** baseline，用于生成图像分类数据选择实验需要的 0/1 mask。

## 方法简介

流程如下：

1. 加载 CIFAR 训练集（无数据增强，只做 `ToTensor + Normalize`）。
2. 用 `torchvision.models.resnet18` 提取每个样本的 embedding（全局池化后、最终 FC 前）。
3. 对 embedding 做 L2 归一化。
4. 对每个类别分别执行 herding 贪心选择：
   - 类内目标保留数 `m_c = round(N_c * keep_ratio)`。
   - 第 `k` 步从未选样本中找一个样本，使加入后已选集合平均特征最接近该类全体样本均值（欧氏距离最小）。
5. 合并各类别选择结果，得到全局 0/1 mask。

## 默认参数

- 数据集：`cifar10`, `cifar100`, `tiny-imagenet`（或 `tiny-imagenet-200`）
- seed：`22, 42, 96`
- keep ratio：`0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2`
- backbone：`resnet18`
- `batch_size=128`
- `num_workers=4`
- 单卡优先：默认 `cuda:0`，无 GPU 时退化到 CPU
- 特征归一化：开启（L2）
- 预训练策略：优先使用 ImageNet 预训练；若下载/加载失败，自动提示并退化为随机初始化

## 运行依赖

- Python 3.9+
- PyTorch
- torchvision
- numpy
- tqdm

## 统一运行脚本

### Tiny-ImageNet 数据准备

默认会在 `--data-root` 下查找：

```text
[--data-root]/tiny-imagenet-200/train
```

例如默认路径：`herding/data/tiny-imagenet-200/train`。


推荐在仓库根目录执行：

```bash
python herding/run_all_herding.py
```

也支持模块方式：

```bash
python -m herding.run_all_herding
```

可选参数示例：

```bash
python herding/run_all_herding.py \
  --data-root herding/data \
  --output-root herding/mask \
  --cache-dir herding/cache
```

## 默认目录（都在 herding/ 下）

脚本默认把数据、中间缓存与输出都放在 `herding/` 下：

- 数据：`herding/data`
- 特征缓存：`herding/cache`
- mask 输出：`herding/mask`

## 输出与命名

mask 保存路径严格为：

```text
[output_root]/[dataset]/[seed]/mask_[cut_ratio].npz
```

按默认参数即：

```text
herding/mask/[dataset]/[seed]/mask_[cut_ratio].npz
```

其中 `cut_ratio = (1 - keep_ratio) * 100`，例如：

- keep 0.9 -> `mask_10.npz`
- keep 0.8 -> `mask_20.npz`
- ...
- keep 0.2 -> `mask_80.npz`

`.npz` 文件至少包含键：

- `mask`: 长度为训练集大小 `N` 的 0/1 数组（整型，`uint8`）

## 缓存说明

为保证 **每个 dataset + seed 只提取一次特征**，脚本默认缓存：

```text
herding/cache/[dataset]_seed[seed]_resnet18.pt
```

重复运行时会优先复用缓存特征，再生成所有 keep ratio 对应 mask。

如需强制重提取，可加 `--disable-cache`。
