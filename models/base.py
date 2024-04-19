"""Base classes for models"""

import os

import torch
import torch.nn as nn

__all__ = ["Activation", "Linear"]


class Activation(nn.Module):
    """Activation Block"""

    def __init__(self, activation):
        """
        Parameters
        ----------
        activation : str
            Activation function

        """
        super(Activation, self).__init__()
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Mish":
            self.activation = nn.Mish()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Identity":
            self.activation = nn.Identity()
        elif activation == "SiLU":
            self.activation = nn.SiLU()
        else:
            raise ValueError("Activation function not implemented.")

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ---------
        x : torch.tensor
            image

        """
        return self.activation(x)


class Linear(nn.Module):
    """Linear Block"""

    def __init__(self, fin, fout,
                 activation="SiLU"):
        """
        Parameters
        ----------
        fin : int
            Input features

        fout : int
            Output fetures

        activation : str, default=SiLU
            Activation function

        """
        super(Linear, self).__init__()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.device = int(os.environ["LOCAL_RANK"])
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.linear = nn.Sequential(
            nn.Linear(fin, fout, device=self.device),
            Activation(activation)
        )

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ---------
        x : torch.tensor
            image

        """
        return self.linear(x)
