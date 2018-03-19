"""
This file holds the functional components of the model.
"""

import numpy as np
import torch
import torch.nn.functional as F

eps = np.finfo(float).eps


def autoregression_nll(z, z_dist, autoregression_bins):
    """
    Returns the autoregression loss associated with a given
    latent representation and its estimated distribution.

    Parameters
    ----------
    z: Variable
        the compressed representation of an input clip.
        Has shape=(batch_size, time_steps, code_length).
    z_dist: Variable
        the distribution over z variables estimated by the
        AND estimator module with autoregression.
        Has shape=(batch_size, autoregression_bins, time_steps, code_length).
    autoregression_bins: int
        The number of bins in which the [0, 1] space is quantized in.

    Returns
    -------
    tuple
        nll: Variable
            the whole batch negative log-likelihood.
        sample_nll: Variable
            the negative log-likelihood for each element of the batch.
    """

    # Apply softmax to the distribution
    z_dist = F.softmax(z_dist, dim=1)

    # Flatten out codes and distributions
    z = z.view(len(z), -1).contiguous()
    z_dist = z_dist.view(len(z), autoregression_bins, -1).contiguous()

    # Turn to logarithm (regularized) and index the correct bins
    z_dist = torch.clamp(z_dist, eps, 1 - eps)
    log_z_dist = torch.log(z_dist)
    index = torch.clamp(torch.unsqueeze(z, dim=1) * autoregression_bins, min=0, max=(autoregression_bins-1)).long()
    selected = torch.gather(log_z_dist, dim=1, index=index)
    selected = torch.squeeze(selected, dim=1)

    # Average out
    nll = - torch.mean(selected)
    sample_nll = - torch.mean(selected, dim=1)

    return nll, sample_nll
