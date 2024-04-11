"""Base classes for models"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Activation", "Convolution",
           "Downsample", "Normalisation",
           "ResNetBlock", "Upsample"]


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


class Convolution(nn.Module):
    """Convolutional Block"""

    def __init__(self, fin, fout,
                 kernel=3, padding=1,
                 stride=1, activation="SiLU",
                 groupnorm=16):
        """
        Parameters
        ----------
        fin : int
            Input features

        fout : int
            Output fetures

        kernel : int or tuple, default=3
            Kernel size

        padding : int, default=1
            Padding size

        stride : int or tuple, default=1
            Stride size

        activation : str, default=SiLU
            Activation function

        groupnorm : int, default=16
            Number of groups for group normalisation

        """
        super(Convolution, self).__init__()
        self.convolution = nn.Sequential(
            Normalisation(fin, groupnorm),
            Activation(activation),
            nn.Conv2d(fin, fout,
                      kernel_size=kernel,
                      padding=padding,
                      stride=stride),
        )

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ---------
        x : torch.tensor
            image

        """
        return self.convolution(x)


class Downsample(nn.Module):
    """Downsampling Block"""

    def __init__(self, features):
        """
        Parameters
        ----------
        features : int
            Number of fetures

        """
        super(Downsample, self).__init__()
        self.downsample = nn.Conv2d(features, features,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ----------
        x : torch.tensor
            image

        Returns
        -------
        h : torch.tensor
            downsampled image

        """
        h = self.downsample(x)
        return h


class Normalisation(nn.Module):
    """Normalisation Block"""

    def __init__(self, features, groups):
        """
        Parameters
        ----------
        features : int
            Features

        groups : int
            Number of groups for group normalisation. If -1 uses BatchNorm2d

        """
        super(Normalisation, self).__init__()
        if groups == -1:
            self.normalisation = nn.BatchNorm2d(features)
        else:
            self.normalisation = nn.GroupNorm(groups, features, eps=1e-6)

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ---------
        x : torch.tensor
            image

        """
        return self.normalisation(x)


class ResNetBlock(nn.Module):
    """Double Convolutional ResNet Block"""

    def __init__(self, fin, fout,
                 kernel=3, padding=1, stride=1,
                 activation="SiLU",
                 groupnorm=16):
        """
        Parameters
        ----------
        fin : int
            Input features

        fout : int
            Output fetures

        kernel : int or tuple, default=3
            Kernel size

        padding : int, default=1
            Padding size

        stride : int or tuple, default=1
            Stride size

        activation : str, default=SiLU
            Activation function

        groupnorm : int, default=16
            Number of groups for group normalisation

        """
        super(ResNetBlock, self).__init__()

        self.convolution = nn.Sequential(
            Normalisation(fin, groupnorm),
            Activation(activation),
            nn.Conv2d(fin, fout,
                      kernel_size=kernel,
                      padding=padding,
                      stride=stride),
            Normalisation(fout, groupnorm),
            Activation(activation),
            nn.Conv2d(fout, fout,
                      kernel_size=kernel,
                      padding=padding,
                      stride=stride),
        )

        if fin != fout:
            self.output = nn.Conv2d(fin, fout,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1)
        else:
            self.output = nn.Identity()

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ---------
        x : torch.tensor
            image

        """
        h = self.convolution(x)
        return self.output(x) + h


class Upsample(nn.Module):
    """Upsampling Block"""

    def __init__(self, fin, fout, method="interpolate"):
        """
        Parameters
        ----------
        fin : int
            Input features

        fout : int
            Output fetures

        method : str, default=interpolation
            Method for upsampling         

        """
        super(Upsample, self).__init__()
        self.method = method
        if method == "interpolate":
            self.upsample = nn.Conv2d(fin, fout,
                                      kernel_size=3,
                                      padding=1)
        else:
            self.upsample = nn.ConvTranspose2d(fin, fout,
                                               kernel_size=2,
                                               stride=2)

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ----------
        x : torch.tensor
            image

        Returns
        -------
        h : torch.tensor
            upsampled image

        """
        if self.method == "interpolate":
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h = self.upsample(x)
        return h
