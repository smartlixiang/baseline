# [ICLR 2025]🎉 A CLIP-Powered Framework for Robust and Generalizable Data Selection

## Introduction
This is the implementation of the clip-powered data selection framework (). In this paper, we propose a novel CLIP-powered data selection framework that leverages multimodal information for more robust and generalizable sample selection. 
You can directly start off using our implementations.

## Getting Started
- Codes support Python3

- Clone this directory and `cd`  into it.
 
`git clone https://github.com/Jackbrocp/clip-powered-data-selection` 

`cd clip-powered-data-selection`

## Updates
- 2025/2/25: Initial release

## Getting Started
### Requirements
- Python 3
- PyTorch 1.6.0
- Torchvision 0.7.0
- Numpy
- CLIP
<!-- Install a fitting Pytorch version for your setup with GPU support, as our implementation  -->

## Usage and Examples 
### Prepare the datasets
Download the datasets (e.g., CIFAR datasets) and put the datasets under the folder ```data/```

### Parameters
```--dataset```, indicates the dataset, by default, ```CIFAR100```

### 1. Fine-tune CLIP's Adapters
This step is to prepare the adapters for CLIP on your own datasets.

Fine-tune linear image and text adapters using InfoNCE loss or contrastive loss and put the adapter in ./adapter_ckpt/{DATASET}/

### 2. Sample Scoring
This step is to pre-calculate the sample scores for selection.
```
python sample_scoring.py --dataset CIFAR100
```

### 3. Selection Optimization 
Optimize the selected datasets w.r.t. specific selection ratios.
```
python optimize_selection.py --dataset CIFAR100
```


## Acknowledge 
Part of our implementation is adopted from the [CLIP](https://github.com/openai/CLIP) repositories.


## Citation
If you find this repository useful in your research, please cite our paper☺️:

```
@article{yang2024clip,
  title={A CLIP-Powered Framework for Robust and Generalizable Data Selection},
  author={Yang, Suorong and Ye, Peng and Ouyang, Wanli and Zhou, Dongzhan and Shen, Furao},
  journal={arXiv preprint arXiv:2410.11215},
  year={2024}
}
```
