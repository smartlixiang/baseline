"""Deprecated helper module kept for compatibility with older imports."""

import numpy as np


def flatten_jacobian(jacobian_tree):
    raise NotImplementedError('JAX jacobian utilities were removed in the PyTorch migration.')


def get_mean_logit_gradients_fn(*_args, **_kwargs):
    raise NotImplementedError('Not used in the PyTorch score pipeline.')


def compute_mean_logit_gradients(*_args, **_kwargs):
    raise NotImplementedError('Not used in the PyTorch score pipeline.')
