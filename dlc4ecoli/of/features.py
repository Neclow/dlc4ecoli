"""Optical flow features"""

import torch

STATS = ("mean", "var", "skew", "kurt")


def get_stats(flow):
    """
    Get optical flow statistics for each frame

    Parameters
    ----------
    flow : torch.Tensor
        Optical flow tensor, N x 2 x H x W

    Returns
    -------
    f_stats : torch.Tensor
        Mean, variance, skewness, kurtosis of the optical flow for each frame
    """
    stat_kwargs = {"dim": -1, "keepdim": True}

    # N x 2 x H x W --> N x H x W --> N x (H x W)
    f_norm = flow.norm(dim=1).flatten(start_dim=1)

    f_mean = f_norm.mean(**stat_kwargs)

    f_diff = f_norm.sub(f_mean)

    f_var = f_diff.pow(2).sum(**stat_kwargs).div(f_diff.shape[-1] - 1)

    z = f_diff.div(f_var.sqrt())

    f_skew = z.pow(3).mean(**stat_kwargs)

    f_kurt = z.pow(4).mean(**stat_kwargs) - 3

    f_stats = torch.cat([f_mean, f_var, f_skew, f_kurt], dim=stat_kwargs["dim"])

    return f_stats
