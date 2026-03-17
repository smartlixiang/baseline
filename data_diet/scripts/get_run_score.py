# python get_run_score.py <ROOT:str> <EXP:str> <RUN:int> <STEP:int> <BATCH_SZ:int> <TYPE:str>

import os
import sys

import numpy as np

from data_diet.data import load_data
from data_diet.scores import compute_scores
from data_diet.utils import load_args, load_model_for_scoring

ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
STEP = int(sys.argv[4])
BATCH_SZ = int(sys.argv[5])
TYPE = sys.argv[6]

run_dir = ROOT + f'/exps/{EXP}/run_{RUN}'
args = load_args(run_dir)
args.load_dir = run_dir
args.ckpt = STEP

_, X, Y, _, _, args = load_data(args)
model, device, _ = load_model_for_scoring(args)
model.eval()
scores = compute_scores(model, device, X, Y, BATCH_SZ, TYPE)

path_name = 'error_l2_norm_scores' if TYPE == 'l2_error' else 'grad_norm_scores'
save_dir = run_dir + f'/{path_name}'
os.makedirs(save_dir, exist_ok=True)
np.save(save_dir + f'/ckpt_{STEP}.npy', scores)
