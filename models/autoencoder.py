"""Autoencoders"""

# pylint: disable=too-many-arguments, unused-argument, import-error, no-name-in-module, consider-using-enumerate, invalid-name
import math

from torch import nn

from models.base import Linear
from models.distribution import GaussianDistribution

__all__ = ["instantiate"]


class Encoder(nn.Module):
    """Enconder"""

    def __init__(self,
                 activation,
                 features,
                 pixels,
                 latent=None):
        """
        Parameters
        ----------
        activation : str
            Activation function

        features : List[int]
            List of features to be generated throughout each layer    

        pixels : int
            Number of pixels in the input image

        latent : int, optional
            Number of features in the latent space. If None then the output is the final feature layer

        """
        super(Encoder, self).__init__()
        self.pixels = pixels

        # Transform the image space
        self.input = nn.Flatten()

        # Compress the input into a lower dimensional space
        self.compression = nn.ModuleList()
        for i in range(0, len(features)):
            fin, fout = self.pixels if i == 0 else features[i - 1], features[i]
            self.compression.append(
                Linear(fin, fout,
                       activation=activation)
            )
            fin = fout

        # Transform the lower dimensional space for the latent space
        self.latent = latent
        if self.latent is not None:
            self.output = Linear(features[-1], latent,
                                 activation="Identity")

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
            compressed space

        """
        h = self.input(x)
        for block in self.compression:
            h = block(h)
        if self.latent is None:
            return h
        z = self.output(h)
        return z


class Decoder(nn.Module):
    """Decoder"""

    def __init__(self,
                 activation,
                 features,
                 pixels,
                 latent=None):
        """
        Parameters
        ----------
        activation : str
            Activation function

        features : List[int]
            List of features to be generated throughout each layer    

        pixels : int
            Number of pixels in the input image

        latent : int, optional
            Number of features in the latent space. If None then the input layer
            considers last feature layer

        """
        super(Decoder, self).__init__()
        self.pixels = pixels

        # Transform the latent space back to the lower dimensional space
        self.latent = latent
        if self.latent is not None:
            self.input = Linear(latent, features[-1], activation=activation)

        # Decompress the latent space into a high dimensional space
        self.decompression = nn.ModuleList()
        for i in reversed(range(0, len(features))):
            fin, fout = features[i], self.pixels if i == 0 else features[i - 1]
            self.decompression.append(
                Linear(fin, fout,
                       activation=activation if i != 0 else "Sigmoid")
            )
            fin = fout

        # Transform back into the image space
        self.output = nn.Unflatten(1, (1, int(math.sqrt(pixels)), int(math.sqrt(pixels))))

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

    def forward(self, z):
        """Apply the forward operation.

        Parameters
        ----------
        z : torch.tensor
            Latent space

        Returns
        -------
        h : torch.tensor
            Image space

        """
        if self.latent is not None:
            h = self.input(z)
        else:
            h = z
        for block in self.decompression:
            h = block(h)
        y = self.output(h)
        return y


class Discriminator(nn.Module):
    """Discriminator"""

    def __init__(self,
                 activation,
                 features,
                 latent):
        """
        Parameters
        ----------
        activation : str
            Activation function

        features : List[int]
            List of features to be generated throughout each layer

        latent : int
            Number of features in the latent space

        """
        super(Discriminator, self).__init__()

        # Transform the latent space back to the lower dimensional space
        self.input = Linear(latent, features[-1], activation=activation)

        # Decompress the latent space into a high dimensional space
        self.decompression = nn.ModuleList()
        for i in reversed(range(0, len(features))):
            fin, fout = features[i], 1 if i == 0 else features[i - 1]
            self.decompression.append(
                Linear(fin, fout,
                       activation=activation if i != 0 else "Identity")
            )
            fin = fout

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

    def forward(self, z):
        """Apply the forward operation.

        Parameters
        ----------
        z : torch.tensor
            Latent space

        Returns
        -------
        h : torch.tensor
            Image space

        """
        h = self.input(z)
        for block in self.decompression:
            h = block(h)
        return h


class AE(nn.Module):
    """Autoencoder"""

    def __init__(self,
                 activation,
                 features,
                 pixels,
                 latent):
        """
        Parameters
        ----------
        activation : str
            Activation function

        features : List[int]
            List of features to be generated throughout each layer    

        pixels : int
            Number of pixels in the input image

        latent : int
            Number of features in the latent space

        """
        super(AE, self).__init__()

        # Encoder layer
        self.encoder = Encoder(activation=activation,
                               features=features,
                               pixels=pixels,
                               latent=latent)

        # Decoder layer
        self.decoder = Decoder(activation=activation,
                               features=features,
                               pixels=pixels,
                               latent=latent)

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


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, pixels,
                 features,
                 latent,
                 activation):
        """
        Parameters
        ----------
        pixels : int
            Number of pixels in the input image

        features : List[int]
            List of features to be generated throughout each layer

        latent : int
            Number of features in the latent space

        activation : str
            Activation function

        """
        super(VAE, self).__init__()

        # Encoder layer
        self.encoder = Encoder(activation=activation,
                               features=features,
                               pixels=pixels)

        # Posterior moments
        self.moments = Linear(features[-1], 2 * latent, activation="Identity")

        # Post Quantised
        self.postQuantised = Linear(latent, features[-1], activation=activation)

        # Decoder layer
        self.decoder = Decoder(activation=activation,
                               features=features,
                               pixels=pixels)

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

    def encode(self, x):
        """Encode the input into a lower dimensional space

        Parameters
        ----------
        x : torch.Tensor
            image

        Returns
        -------
        posterior : GaussianDistribution
            posterior Gaussian distribution

        """
        h = self.encoder(x)
        moments = self.moments(h)
        posterior = GaussianDistribution(moments)
        return posterior

    def decode(self, z):
        """Decode the latent space into the image space

        Parameters
        ----------
        z : torch.Tensor
            latent variable

        Returns
        -------
        y : torch.Tensor
            reconstructed image

        """
        h = self.postQuantised(z)
        y = self.decoder(h)
        return y

    def forward(self, x, method="full"):
        """Apply the forward operation.

        Parameters
        ----------
        x : torch.tensor
            image

        method : str
            Whether to do the full pass or just the encoder pass

        Returns
        -------
        y : torch.tensor, optional
            reconstructed image only on the full pass

        posterior : GaussianDistribution, optional
            Posterior Gaussian distribution

        """
        if method == "encode":
            posterior = self.encode(x)
            return posterior
        elif method == "decode":
            y = self.decode(x)
            return y
        elif method == "full":
            posterior = self.encode(x)
            z = posterior.sample()
            y = self.decode(z)
            return y, posterior
        else:
            raise ValueError("Method not implemented")


def instantiate(activation=None,
                architecture=None,
                features=None,
                latent=None,
                pixels=None,
                ** kwargs
                ):
    """Create an instance of the model.

    Parameters
    ----------
    activation : str
        Activation function

    architecture : str
        Architecture to be used: UNet, MGUnet and VAE

    features : int
        List of features to be generated throughout the compressive layers

    latent : int
        Number of features in the latent space

    pixels : int
        Number of pixels in the input image

    Returns
    -------
    model : object(s)
        Model

    """
    if architecture == "AE":
        model = AE(activation=activation,
                   features=features,
                   latent=latent,
                   pixels=pixels)
        return model

    if architecture == "VAE":
        model = VAE(activation=activation,
                    features=features,
                    latent=latent,
                    pixels=pixels)
        return model

    if architecture == "AAE":
        encoder = Encoder(activation=activation,
                          features=features,
                          latent=latent,
                          pixels=pixels)

        decoder = Decoder(activation=activation,
                          features=features,
                          latent=latent,
                          pixels=pixels)

        discriminator = Discriminator(activation=activation,
                                      features=features,
                                      latent=latent)
        return encoder, decoder, discriminator

    raise ValueError("Architecture not implemented.")


# if __name__ == "__main__":
#     import torch
#     import os

#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         device = int(os.environ["LOCAL_RANK"])
#     elif torch.cuda.is_available():
#         device = 'cuda'
#     elif torch.backends.mps.is_available():
#         device = 'mps'
#     else:
#         device = 'cpu'

#     encoder, decoder, discriminator = instantiate(activation="ReLU",
#                                                   architecture="AAE",
#                                                   features=(512, 256),
#                                                   latent=2,
#                                                   pixels=784)
#     encoder = encoder.to(device=device)
#     decoder = decoder.to(device=device)
#     discriminator = discriminator.to(device=device)
#     sample = torch.randn(1, 28, 28).to(device=device)

#     z = encoder(sample)
#     y = decoder(z)
#     pred = discriminator(z)
