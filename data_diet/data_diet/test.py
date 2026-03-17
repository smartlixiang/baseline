import torch

from .data import test_batches
from .metrics import accuracy, cross_entropy_loss


def test(state, X, Y, batch_size, device):
    state.model.eval()
    loss, acc, N = 0.0, 0.0, X.shape[0]
    with torch.no_grad():
        for n, x, y in test_batches(X, Y, batch_size):
            xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y).to(device=device, dtype=torch.float32)
            logits = state.model(xb)
            loss += cross_entropy_loss(logits, yb).item() * n
            acc += accuracy(logits, yb).item() * n
    return loss / N, acc / N
