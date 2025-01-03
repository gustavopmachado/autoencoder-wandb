"""Script for training"""

# pylint: disable=import-error, no-name-in-module, redefined-outer-name, consider-using-f-string

import os

import torch
import yaml
from torch.distributed import destroy_process_group

import wandb
from models.autoencoder import instantiate
from utils.settings import ddp
from utils.training import AAETrainer, AETrainer, VAETrainer


def run(hyperparameters):
    """Perform the training of a model and logged via wandb.
    A dictionary is used as a single argument for wandb, however its
    keys and respective descriptions are listed below as paramters.

    Parameters
    ----------
    activation : str
        Activation function. Available is "LeakyReLU", "ReLU",
        "Tanh", "Mish" and "Sigmoid"

    architecture : str
        Architecture to be used in the model. Values are AE, VAE

    batch : int
        Batch size

    betas : tuple
        Adam's betas

    dataset : str
        Name of the dataset in data/

    epochs : int
        Training epochs

    eta : float, optional
        Kullback–Leibler divergence regularization factor

    features : List[int]
        List of features to be generated throughout the compressive layers

    latent : int, optional
        Number of features in the latent space

    lr : float
        Learning rate

    pixels : int
        Number of pixels in the input image

    Returns
    -------
        Trained model

    """
    # Initialise DDP
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ddp()

    # Initialise wandb for logging
    args = {"project": "autoencoder-wandb", "config": hyperparameters}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        args.update({"name": "GPU: " + str(int(os.environ["LOCAL_RANK"]))})
    with wandb.init(**args):
        config = wandb.config

        # Set up training
        if config.get("architecture") == "AE":
            model = instantiate(**config)
            trainer = AETrainer(model, **config)
        elif config.get("architecture") == "VAE":
            model = instantiate(**config)
            trainer = VAETrainer(model, **config)
        elif config.get("architecture") == "AAE":
            encoder, decoder, discriminator = instantiate(**config)
            trainer = AAETrainer(encoder, decoder, discriminator, **config)
        else:
            raise ValueError("Training not implemented for {}".format(config.get("architecture")))

        # Traning
        model = trainer.run()

    # DDP tear down
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        destroy_process_group()

    return model


if __name__ == "__main__":

    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str,
                        help='architecture to be used in the model',
                        required=True)
    parser.add_argument('--dataset', type=str,
                        help='name of the dataset in data/',
                        required=True)
    parser.add_argument('--pixels', type=int, default=784,
                        help='total number of pixels per image')
    args, unknown = parser.parse_known_args()

    # Training settings
    settings = dict(
        architecture=args.architecture,
        dataset=args.dataset,
        pixels=args.pixels
    )

    # Load harameters for training
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(root, "experiments", settings.get("architecture"), "config.yaml")
    with open(path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    hyperparameters = {}
    for block in config.values():
        hyperparameters.update(**block)
    hyperparameters.update(settings)

    # Training
    model = run(hyperparameters)
