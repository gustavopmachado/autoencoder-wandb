"""Autoencoders"""

# pylint: disable=consider-using-enumerate, invalid-name, too-many-arguments, unused-argument, protected-access, line-too-long
from torch import nn

from models.base import (Convolution, Downsample, ResNetBlock, Upsample)

__all__ = ["AE", "instantiate"]


class Encoder(nn.Module):
    """Convolutional Enconder"""

    def __init__(self, channels,
                 features,
                 latent,
                 resnet,
                 activation,
                 groupnorm,
                 double):
        """
        Parameters
        ----------
        channels : int
            Input image channels

        features : List[int]
            List of features to be generated throughout each layer

        latent : int
            Number of features in the latent space

        resnet : int
            Number of ResNet blocks per layer

        activation : str
            Activation function

        groupnorm : int
            Number of groups for group normalisation

        double : bool
            Whether to double the size of the latent features

        """
        super(Encoder, self).__init__()

        # Map input channels to initial features and keep the initial resolution
        self.input = nn.Conv2d(channels, features[0],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        # Compress the input into a lower dimensional space
        self.compression = nn.ModuleList()
        for i in range(0, len(features)):
            fin, fout = features[i - 1] if i > 0 else features[i], features[i]
            for _ in range(resnet):
                self.compression.append(
                    ResNetBlock(fin, fout,
                                activation=activation,
                                groupnorm=groupnorm)
                )
                fin = fout
            if i < len(features) - 1:
                self.compression.append(
                    Downsample(fout)
                )

        # Transform the lower dimensional space for the latent space
        self.output = Convolution(features[-1],
                                  2 * latent if double else latent,
                                  activation=activation,
                                  groupnorm=groupnorm)

        # Initialize the weights ~ N(0, 0.02)
        self.init_parameters()

    def init_parameters(self, mean=0, std=0.02):
        """Force initialization of parameters according
           to Radford et. al (2015).

        Parameters
        ----------
        mean : float, default=0
            Mean for parameters initialization

        std : float, default=0.02
            Standard deviation for parameters initialization

        """
        for weights in self.parameters():
            weights.data.normal_(mean, std)

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ----------
        x : torch.tensor
            image

        Returns
        -------
        h : torch.tensor
            latent space

        """
        h = self.input(x)
        for block in self.compression:
            h = block(h)
        h = self.output(h)
        return h


class Decoder(nn.Module):
    """Convolutional Decoder """

    def __init__(self, channels,
                 features,
                 latent,
                 resnet,
                 activation,
                 groupnorm):
        """
        Parameters
        ----------
        channels : int
            Input image channels

        features : List[int]
            List of features to be generated throughout each layer

        latent : int
            Number of features in the latent space

        resnet : int
            Number of ResNet blocks per layer

        activation : str
            Activation function

        groupnorm : int
            Number of groups for group normalisation

        """
        super(Decoder, self).__init__()

        # Map latent features to reverse the compression
        self.input = nn.Conv2d(latent, features[-1],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        # Decompress the latent space into a high dimensional space
        self.decompression = nn.ModuleList()
        for i in reversed(range(0, len(features))):
            fin, fout = features[i + 1] if i < len(features) - 1 else features[i], features[i]
            for _ in range(resnet + 1):
                self.decompression.append(
                    ResNetBlock(fin, fout,
                                activation=activation,
                                groupnorm=groupnorm)
                )
                fin = fout
            if i > 0:
                self.decompression.append(
                    Upsample(fout, fout)
                )

        # Transform into the image space
        self.output = Convolution(features[0],
                                  channels,
                                  activation=activation,
                                  groupnorm=groupnorm)

        # Initialize the weights ~ N(0, 0.02)
        self.init_parameters()

    def init_parameters(self, mean=0, std=0.02):
        """Force initialization of parameters according
           to Radford et. al (2015).

        Parameters
        ----------
        mean : float, default=0
            Mean for parameters initialization

        std : float, default=0.02
            Standard deviation for parameters initialization

        """
        for weights in self.parameters():
            weights.data.normal_(mean, std)

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ----------
        x : torch.tensor
            Latent space

        Returns
        -------
        h : torch.tensor
            Image space

        """
        h = self.input(x)
        for block in self.decompression:
            h = block(h)
        h = self.output(h)
        return h


class AE(nn.Module):
    """Convolutional Autoencoder"""

    def __init__(self, channels,
                 features,
                 latent,
                 resnet,
                 activation,
                 groupnorm):
        """
        Parameters
        ----------
        channels : int
            Input image channels

        features : List[int]
            List of features to be generated throughout each layer

        latent : int
            Number of features in the latent space

        resnet : int
            Number of ResNet blocks per layer

        activation : str
            Activation function

        groupnorm : int
            Number of groups for group normalisation

        """
        super(AE, self).__init__()

        # Encoder layer
        self.encoder = Encoder(channels,
                               features,
                               latent,
                               resnet,
                               activation=activation,
                               groupnorm=groupnorm,
                               double=True)

        # Decoder layer
        self.decoder = Decoder(channels,
                               features,
                               latent,
                               resnet,
                               activation=activation,
                               groupnorm=groupnorm)

        # Initialize the weights ~ N(0, 0.02)
        self.init_parameters()

    def init_parameters(self, mean=0, std=0.02):
        """Force initialization of parameters according
           to Radford et. al (2015).

        Parameters
        ----------
        mean : float, default=0
            Mean for parameters initialization

        std : float, default=0.02
            Standard deviation for parameters initialization

        """
        for weights in self.parameters():
            weights.data.normal_(mean, std)

    def forward(self, x):
        """Apply the forward operation.

        Parameters
        ----------
        x : torch.tensor
            image

        Returns
        -------
        y : torch.tensor
            reconstructed image

        """
        z = self.encoder(x)
        y = self.decoder(z)
        return y


def instantiate(activation=None,
                architecture=None,
                channels=None,
                features=None,
                groupnorm=None,
                latent=None,
                resnet=None,
                **kwargs
                ):
    """Create an instance of the model.

    Parameters
    ----------
    activation : str
        Activation function

    architecture : str
        Architecture to be used: UNet, MGUnet and VAE

    channels : int
        Image channel

    connections : List[int]
            List of features in which skip connections will be used

    depth : int, optional
        Depth of the semi-reconstruction layer in relation to the first
        level or surface. Value must be greater than 0. Example, if depth is 1,
        then semi-reconstruction will be done up to the second layer of the architecture

    features : int
        List of features to be generated throughout the compressive layers

    groupnorm : int
        Number of groups for group normalisation

    latent : int, optional
        Number of features in the latent space for the VAE

    resnet : int
        Number of ResNet blocks per layer

    Returns
    -------
    model : object(s)
        Model

    """
    #  Instantiate the model
    if architecture == "AE":
        model = AE(channels=channels,
                   features=features,
                   latent=latent,
                   resnet=resnet,
                   activation=activation,
                   groupnorm=groupnorm)
        return model
    else:
        raise ValueError("Architecture not implemented.")
