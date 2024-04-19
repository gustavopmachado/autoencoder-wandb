"""Probabilistic distributions"""

# pylint: disable=invalid-name, line-too-long

import os

import torch

__all__ = ["GaussianDistribution"]


class GaussianDistribution():
    """Latent space represented by a Gaussian distribution"""

    def __init__(self, x):
        """"
        Parameters
        ----------
        x : torch.tensor of (batch, 2 * dimension, ..., ...)
            Compressed data with mean and log(variance) embedded as channels

        Note
        ----
        Implementation according to the original paper on Latent Diffusion Models:
        [1] https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/distributions/distributions.py#L24

        """
        # General parameters
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.device = int(os.environ["LOCAL_RANK"])
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Posterior moments
        self.mean, self.logvar = torch.chunk(x, 2, dim=1)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        """Sampling from latent space"""
        z = self.mean + self.std * torch.randn_like(self.std, device=self.device)
        return z

    def mode(self):
        """Sampling from latent space using the mean"""
        return self.mean

    def kl(self):
        """Kullback–Leibler Divergence

        Returns
        -------
            KL divergence

        """
        return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                               dim=[1])
