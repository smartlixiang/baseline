#!/bin/bash
FILE=$1
shift

python Eval.py --file ${FILE} "$@"

# ./eval.sh pretrained-rescale_mask_best-acc_40_1023_resnet18.pt --device x --seed x