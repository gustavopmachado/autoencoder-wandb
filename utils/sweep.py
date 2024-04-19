"""Script for training"""

# pylint: disable=import-error, no-name-in-module, consider-using-f-string, redefined-outer-name

import os
from functools import partial

import torch
import yaml
from torch.distributed import destroy_process_group

import wandb
from models.autoencoder import instantiate
from utils.settings import ddp
from utils.training import AAETrainer, AETrainer, VAETrainer


def run(config):
    """Perform the training of a model and logged via wandb.
    A dictionary is used as a single argument for wandb, however its
    keys and respective descriptions are listed below as paramters.

    Parameters
    ----------
    activation : str
        Activation function. Available is "LeakyReLU", "ReLU",
        "Tanh", "Mish" and "Sigmoid"

    batch : int
        Batch size

    betas : tuple
        Adam's betas

    dataset : str
        Name of the datafolder in data/

    epochs : int
        Training epochs

    eta : float, optional
        Kullback–Leibler divergence regularization factor

    features : int
        List of features to be generated throughout the compressive layers

    latent : int, optional
        Number of features in the latent space

    lr : float
        Learning rate

    pixels : int
        Number of pixels in the input image

    """
    # Initialise DDP
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ddp()

    # Initialise wandb for logging
    with wandb.init(config=config):
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


if __name__ == "__main__":

    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str,
                        help='architecture to be used in the model',
                        required=True)
    parser.add_argument("--betas", nargs="+", type=float,
                        help='adam momentum parameters',
                        required=True)
    parser.add_argument('--count', type=int,
                        help='number of sweep config trials to try',
                        required=True)
    parser.add_argument('--dataset', type=str,
                        help='name of the dataset in data/',
                        required=True)
    parser.add_argument('--epochs', type=int,
                        help='training epochs',
                        required=True)
    parser.add_argument('--latent', type=int,
                        help='number of features in the latent space',
                        required=True)
    parser.add_argument('--logfreq', type=int,
                        help='frequency to log with wandb',
                        required=True)
    parser.add_argument('--pixels', type=int, default=784,
                        help='total number of pixels per image')
    args, unknown = parser.parse_known_args()

    # Training settings
    settings = dict(
        architecture=args.architecture,
        betas=args.betas,
        dataset=args.dataset,
        epochs=args.epochs,
        latent=args.latent,
        logfreq=args.logfreq,
        pixels=args.pixels,
        plotfreq=-1,
    )

    # Load hyperparameters for sweep
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(root, "experiments", settings.get("architecture"), "sweep.yml")
    with open(path, encoding="utf-8") as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    # Set up the sweep
    sweepid = wandb.sweep(hyperparameters, project="autoencoder-wandb")

    # Guarantees a cleaner sweep's UI
    hyperparameters.update(settings)

    # Run the sweep
    wandb.agent(sweepid, function=partial(run, hyperparameters),
                count=args.count)
