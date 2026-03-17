# python get_mean_score.py <ROOT:str> <EXP:str> <N_RUNS:int> <STEP:int> <TYPE:str>

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

ROOT = sys.argv[1]
EXP = sys.argv[2]
N_RUNS = int(sys.argv[3])
STEP = int(sys.argv[4])
TYPE = sys.argv[5]

if TYPE == 'l2_error':
  path_name = 'error_l2_norm_scores'
elif TYPE == 'grad_norm':
  path_name = 'grad_norm_scores'
else:
  path_name = 'forget_scores'

exp_dir = ROOT + f'/exps/{EXP}'
scores = []
for run in range(N_RUNS):
  load_path = exp_dir + f'/run_{run}/{path_name}/ckpt_{STEP}.npy'
  scores.append(np.load(load_path))
scores = np.stack(scores).mean(0)

save_dir = exp_dir + f'/{path_name}'
save_path = save_dir + f'/ckpt_{STEP}.npy'
if not os.path.exists(save_dir): os.makedirs(save_dir)
np.save(save_path, scores)
