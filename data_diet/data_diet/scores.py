import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _to_torch_batch(X, Y, device):
    xb = torch.from_numpy(X).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    yb = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
    return xb, yb


def _el2n_scores(model, X, Y, device):
    xb, yb = _to_torch_batch(X, Y, device)
    with torch.no_grad():
        probs = F.softmax(model(xb), dim=-1)
        return torch.linalg.norm(probs - yb, dim=-1).cpu().numpy()


def _grand_scores(model, X, Y, device):
    xb, yb = _to_torch_batch(X, Y, device)
    targets = yb.argmax(dim=-1)
    try:
        from torch.func import functional_call, grad, vmap

        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def loss_single(params_, buffers_, x, y):
            logits = functional_call(model, (params_, buffers_), (x.unsqueeze(0),))
            return F.cross_entropy(logits, y.unsqueeze(0))

        per_grad = vmap(grad(loss_single), in_dims=(None, None, 0, 0))(params, buffers, xb, targets)
        sq = torch.zeros(xb.shape[0], device=device)
        for g in per_grad.values():
            sq += g.reshape(g.shape[0], -1).pow(2).sum(dim=1)
        return torch.sqrt(sq).cpu().numpy()
    except Exception:
        scores = []
        model.zero_grad(set_to_none=True)
        for i in range(xb.shape[0]):
            logits = model(xb[i:i + 1])
            loss = F.cross_entropy(logits, targets[i:i + 1])
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)
            gnorm = torch.sqrt(sum((g.pow(2).sum() for g in grads))).item()
            scores.append(gnorm)
        return np.array(scores, dtype=np.float32)


def compute_scores(model, device, X, Y, batch_sz, score_type):
    n = X.shape[0]
    out = []
    for start in tqdm(range(0, n, batch_sz), desc=f"score:{score_type}", dynamic_ncols=True):
        end = min(start + batch_sz, n)
        if score_type == 'l2_error':
            out.append(_el2n_scores(model, X[start:end], Y[start:end], device))
        elif score_type == 'grad_norm':
            out.append(_grand_scores(model, X[start:end], Y[start:end], device))
        else:
            raise NotImplementedError(score_type)
    return np.concatenate(out, axis=0)
