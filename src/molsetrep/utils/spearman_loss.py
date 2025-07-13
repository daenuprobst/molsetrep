import torch
from fast_soft_sort.pytorch_ops import soft_rank


def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()


def _find_repeats(data):
    temp = data.detach().clone()
    temp = temp.sort()[0]

    change = torch.cat(
        [torch.tensor([True], device=temp.device), temp[1:] != temp[:-1]]
    )
    unique = temp[change]
    change_idx = torch.cat(
        [torch.nonzero(change), torch.tensor([[temp.numel()]], device=temp.device)]
    ).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]


def _rank_data(data):
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank


def spearman_loss(pred, gt, regularization_strength=1.0, regularization="l2"):
    pred = pred.unsqueeze(0)
    gt = gt.unsqueeze(0)

    assert pred.device == gt.device
    assert pred.shape == gt.shape
    assert pred.shape[0] == 1
    assert pred.ndim == 2

    device = pred.device

    soft_pred = soft_rank(
        pred.cpu(),
        regularization_strength=regularization_strength,
        regularization=regularization,
    ).to(device)

    soft_true = _rank_data(gt.squeeze(0)).to(device)
    preds_diff = soft_pred - soft_pred.mean()
    target_diff = soft_true - soft_true.mean()

    cov = (preds_diff * target_diff).mean()
    preds_std = torch.sqrt((preds_diff * preds_diff).mean())
    target_std = torch.sqrt((target_diff * target_diff).mean())

    spearman_corr = cov / (preds_std * target_std + 1e-6)
    return -spearman_corr
