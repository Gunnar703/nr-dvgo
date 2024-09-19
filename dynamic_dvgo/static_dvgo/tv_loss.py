import torch

def total_variation(v: torch.Tensor, mask=None):
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    if mask is not None:
        tv2 = tv2[mask[:, :, :-1] & mask[:, :, 1:]]
        tv3 = tv3[mask[:, :, :, :-1] & mask[:, :, :, 1:]]
        tv4 = tv4[mask[:, :, :, :, :-1] & mask[:, :, :, :, 1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3