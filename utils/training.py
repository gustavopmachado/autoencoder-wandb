"""Utility methods for model training and evaluation"""

# pylint: disable=redefined-outer-name, disallowed-name, line-too-long, unused-argument, no-member, not-callable, super-init-not-called, invalid-name

import os
import random
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import progressbar
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

import wandb
from utils.dataset import data

PIL.Image.MAX_IMAGE_PIXELS = None

__all__ = ["AETrainer", "VAETrainer", "AAETrainer"]


class BaseTrainer(ABC):
    """Base class for training"""

    def __init__(self, model, batch=None, betas=None,
                 dataset=None, epochs=None, logfreq=None,
                 lr=None, plotfreq=None):
        """
        Parameters
        ----------
        model : object
            Torch model

        batch : int
            Batch size

        betas : tuple
            Adam's betas

        dataset : str
            Name of the dataset

        epochs : int
            Training epochs

        logfreq : int
            Frequency in which the training will be logged

        lr : float
            Adam's Learning rate

        plotfreq : int
            Frequency in which the validation results will be plotted throughout training

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

        # Dataset
        self.dataset = dataset
        self.batch = batch
        self.train_data, self.valid_data = data(self.dataset,
                                                self.batch)

        # Model ojects
        self.model = model.to(self.device)
        self.architecture = type(model).__name__

        # Optimiser
        self.optimiser = Adam(self.model.parameters(), lr=lr, betas=betas)

        # Loss
        self.criterion = MSELoss

        # Training
        self.epochs = epochs
        self.epochsrun = 0

        # Logging
        self.logfreq = logfreq
        self.plotfreq = plotfreq

        # Initialise DDP
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[self.device])

    @abstractmethod
    def train(self, data):
        """Training step to be implemented"""

    @abstractmethod
    def validate(self, data, epoch):
        """Validation step to be implemented"""

    @staticmethod
    def setup(seed=42):
        """Setup the training process"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            torch.mps.empty_cache()
        if os.getenv("DEBUG_MODE", 'False').lower() in ('true', '1', 't'):
            torch.autograd.set_detect_anomaly(True)

    def run(self):
        """Returns a trained model"""

        # Setup the training
        self.setup()

        # Set the logging
        wandb.watch(self.model, self.criterion(),
                    log=None, log_freq=self.logfreq)
        log = {"Epochs": [], "Training Loss": [], "Validation Loss": []}

        # Train the model
        with progressbar.ProgressBar(min_value=self.epochsrun, max_value=self.epochs) as bar:
            for epoch in range(self.epochsrun, self.epochs):

                # Logging
                bar.update(epoch + 1)
                log['Epochs'].append(epoch)

                # Training step
                train_loss = self.train(self.train_data)
                log['Training Loss'].append(train_loss)

                # Validation step
                valid_loss, valid_log = self.validate(self.valid_data, epoch)
                log['Validation Loss'].append(valid_loss)

                # Logging
                if epoch % self.logfreq == 0 or epoch == self.epochs - 1:
                    log.update(valid_log)
                    self.logging(epoch, log)

        # Save the model
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        #     if self.device == int(os.environ["MAIN_RANK"]):
        #         self.save(epoch)
        # else:
        #     self.save(epoch)

        return self.model

    def save(self, epoch):
        """Save the current snapshot of the training model.

        Parameters
        ----------
        epochs : int
            Training epochs

        """
        state = (self.model.module.state_dict()
                 if torch.distributed.is_available() and torch.distributed.is_initialized()
                 else self.model.state_dict())
        snapshot = {
            "MODEL_STATE": state,
            "EPOCHS_RUN": epoch,
        }
        name = f"{self.architecture}-{epoch}.pt"
        torch.save(snapshot, os.path.join(self.savepath, name))

    def load(self):
        """Load last saved snapshot of training model."""
        snapshot = torch.load(self.savepath,
                              map_location=f"cuda:{self.device}")
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochsrun = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at epoch {self.epochsrun}")

    def logging(self, epoch, log):
        """Log training to wandb and to console.

        Parameters
        ----------
        log : dict
            Dictionary containing data to be logged

        """
        if self.plotfreq != -1 and (epoch % self.plotfreq == 0 or epoch == self.epochs - 1):
            fig = self.plot(log)
            wandb.log({"Training Loss": log.get("Training Loss")[-1],
                       "Validation Loss": log.get("Validation Loss")[-1],
                       "Validation Results": fig},
                      step=log.get("Epochs")[-1])
            plt.close(fig)
        else:
            wandb.log({"Training Loss": log.get("Training Loss")[-1],
                       "Validation Loss": log.get("Validation Loss")[-1]},
                      step=log.get("Epochs")[-1])

    def plot(self, log):
        """Visualise the latent space and results.

        Paramters
        ---------
        log : dict
            Dictionary containing log information

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure with the visualisation

        """
        matplotlib.rcParams.update({'font.size': 8})
        fig = plt.figure(figsize=(12, 2))
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.,
                                  width_ratios=[0.3, 0.7])

        for i in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(1 if i == 0 else 2,
                                                     1 if i == 0 else 10,
                                                     subplot_spec=outer[i],
                                                     wspace=0.2, hspace=0)
            # Visualise the latent space
            if i == 0:
                ax = plt.Subplot(fig, inner[0])
                for image, label in self.valid_data:
                    if self.architecture == "VAE":
                        posterior = self.model(image.to(self.device), method="encode")
                        z = posterior.sample().to('cpu').detach().numpy()
                    else:
                        z = self.model.encoder(image.to(self.device)).to('cpu').detach().numpy()
                    im = ax.scatter(z[:, 0], z[:, 1], s=3.0, c=label, cmap='tab10', alpha=.65)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title("Latent Space")
                fig.add_subplot(ax)

            # Visualise the results
            if i == 1:
                for j, _ in enumerate(log["Temporal"]["labels"]):
                    ax = plt.Subplot(fig, inner[j])
                    im = ax.imshow(log["Temporal"]["images"][j].cpu().squeeze())
                    ax.set_yticks([])
                    ax.set_xticks([])
                    if j == 0:
                        ax.set_ylabel("Original")
                    fig.add_subplot(ax)

                    ax = plt.Subplot(fig, inner[j + len(log["Temporal"]["labels"])])
                    im = ax.imshow(log["Temporal"]["recon"][j].cpu().squeeze())
                    ax.set_yticks([])
                    ax.set_xticks([])
                    if j == 0:
                        ax.set_ylabel("Recon")
                    fig.add_subplot(ax)

        fig.get_axes()[0].remove()
        return fig

    def select(self, recon, images, labels, log):
        """Select the individuals on validation set in which
        best fit an equidistant temporal mesh.

        Parameters
        ----------
        images : torch.tensor
            True data

        log : dict
            Dictionary containing log information

        recon : torch.tensor
            Reconstructed data

        labels : torch.tensor
            Labels

        Returns
        -------
            Dictionary containing the updated log information

        """
        empty = torch.tensor([], requires_grad=False, device=self.device)

        # Select the 1st object for each class
        indices = torch.tensor([], dtype=torch.int, requires_grad=False, device=self.device)
        for target in labels.unique():
            indices = torch.cat((indices,
                                 (labels == target).nonzero(as_tuple=True)[0][0].reshape(1).to(self.device)),
                                0)

        # Calculate the new losses
        loss = torch.mean(self.criterion(reduction='none')(recon, images),
                          dim=[1, 2, 3]).squeeze()

        # Update the log
        log.update(
            {"Temporal": {"loss": torch.cat([log.get("Temporal", {}).get("loss", empty), loss])[indices],
                          "images": torch.cat([log.get("Temporal", {}).get("images", empty), images])[indices],
                          "recon": torch.cat([log.get("Temporal", {}).get("recon", empty), recon])[indices],
                          "labels": labels.to(self.device)[indices]}}
        )
        return log


class AETrainer(BaseTrainer):
    """Autoencoder training class"""

    def __init__(self, model, batch=None, betas=None,
                 dataset=None, epochs=None,
                 logfreq=None, lr=None, plotfreq=None,
                 **kwargs):
        """
        Parameters
        ----------
        model : object
            Torch model

        batch : int
            Batch size

        betas : tuple
            Adam's betas

        dataset : str
            Name of the datafolder in data/

        epochs : int
            Training epochs

        logfreq : int
            Frequency in which the training will be logged

        lr : float
            Adam's Learning rate

        """
        super(AETrainer, self).__init__(model,
                                        batch=batch,
                                        betas=betas,
                                        dataset=dataset,
                                        epochs=epochs,
                                        logfreq=logfreq,
                                        lr=lr,
                                        plotfreq=plotfreq)

    def train(self, data):
        """Training step.

        Parameters
        ----------
        data : DataLoader
            Training dataset

        epoch : int
            Training epoch

        Returns
        -------
            Training Loss

        """
        total = 0.
        self.model.train()
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        for images, _ in data:
            self.optimiser.zero_grad(set_to_none=True)
            images = images.to(self.device)
            recon = self.model(images)
            loss = self.criterion()(recon, images)
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimiser)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if torch.cuda.is_available():
                scaler.step(self.optimiser)
                scaler.update()
            else:
                self.optimiser.step()
            total += loss.item() * images.size(0)
        total /= len(data.dataset)
        return total

    def validate(self, data, epoch):
        """Validation step.

        Parameters
        ----------
        data : DataLoader
            Validation dataset

        epoch : int
            Training epoch

        Returns
        -------
            Validation Loss

        """
        total, log = 0., {}
        self.model.eval()
        with torch.no_grad():
            for images, labels in data:
                images = images.to(self.device)
                recon = self.model(images)
                loss = self.criterion()(recon, images)
                total += loss.item() * images.size(0)
                if epoch % self.logfreq == 0 and not log:
                    log = self.select(recon, images, labels, log)
        total /= len(data.dataset)
        return total, log


class VAETrainer(BaseTrainer):
    """Varational Autoencoder training class"""

    def __init__(self, model, batch=None, betas=None,
                 dataset=None, epochs=None, eta=None,
                 logfreq=None, lr=None, plotfreq=None,
                 **kwargs):
        """
        Parameters
        ----------
        model : object
            Torch model

        batch : int
            Batch size

        betas : tuple
            Adam's betas

        dataset : str
            Name of the datafolder in data/

        epochs : int
            Training epochs

        eta : float
            Kullback–Leibler Divergence regularisation

        logfreq : int
            Frequency in which the training will be logged

        lr : float
            Adam's Learning rate

        """
        super(VAETrainer, self).__init__(model,
                                         batch=batch,
                                         betas=betas,
                                         dataset=dataset,
                                         epochs=epochs,
                                         logfreq=logfreq,
                                         lr=lr,
                                         plotfreq=plotfreq)

        # Training
        self.eta = eta

    def train(self, data):
        """Training step.

        Parameters
        ----------
        data : DataLoader
            Training dataset

        epoch : int
            Training epoch

        Returns
        -------
            Training Loss

        """
        total = 0.
        self.model.train()
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        for images, _ in data:
            self.optimiser.zero_grad(set_to_none=True)
            images = images.to(self.device)
            recon, posterior = self.model(images)
            # loss = (self.criterion(reduction="sum")(recon, images) +
            #         self.eta * posterior.kl().sum())
            loss = (self.criterion()(recon, images) +
                    self.eta * posterior.kl().mean())
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimiser)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if torch.cuda.is_available():
                scaler.step(self.optimiser)
                scaler.update()
            else:
                self.optimiser.step()
            total += loss.item() * images.shape[0]
        total /= len(data.dataset)
        return total

    def validate(self, data, epoch):
        """Validation step.

        Parameters
        ----------
        data : DataLoader
            Validation dataset

        epoch : int
            Training epoch

        Returns
        -------
            Validation Loss

        """
        total, log = 0., {}
        self.model.eval()
        with torch.no_grad():
            for images, labels in data:
                images = images.to(self.device)
                recon, posterior = self.model(images)
                # loss = (self.criterion(reduction="sum")(recon, images) +
                #         self.eta * posterior.kl().sum())
                loss = (self.criterion()(recon, images) +
                        self.eta * posterior.kl().mean())
                total += loss.item() * images.shape[0]
                if epoch % self.logfreq == 0 and not log:
                    log = self.select(recon, images, labels, log)
        total /= len(data.dataset)
        return total, log


class AAETrainer(BaseTrainer):
    """Class to perform adversarial training"""

    def __init__(self, encoder, decoder, discriminator,
                 batch=None, betas=None, dataset=None,
                 epochs=None, latent=None, logfreq=None,
                 lr=None, advlr=None, plotfreq=None, **kwargs):
        """
        Parameters
        ----------
        encoder : object
            Encoder model

        decoder : object
            Encoder model

        discriminator : object
            Encoder model

        batch : int
            Batch size

        betas : tuple
            Adam's betas

        dataset : str
            Name of the dataset in data/

        epochs : int
            Training epochs

        latent : int
            Number of features in Autoencoder's latent space

        logfreq : int
            Frequency in which the training will be logged

        lr : float
            Adam's Learning rate for the Encoder and Decoder

        advlr : float
            Adam's Learning rate for the Generator and Discriminator

        plotfreq : int
            Frequency in which the validation results will be plotted throughout training

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
        self.architecture = "AAE"

        # Model ojects
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # Optimiser
        self.encoder_optimiser = Adam(self.encoder.parameters(),
                                      lr=lr,
                                      betas=betas)
        self.decoder_optimiser = Adam(self.decoder.parameters(),
                                      lr=lr,
                                      betas=betas)
        self.generator_optimiser = Adam(self.encoder.parameters(),
                                        lr=advlr,
                                        betas=betas)
        self.discriminator_optimiser = Adam(self.discriminator.parameters(),
                                            lr=advlr,
                                            betas=betas)

        # Loss
        self.autoencoder_criterion = MSELoss
        self.adversarial_criterion = BCEWithLogitsLoss

        # Dataset parameters
        self.dataset = dataset
        self.batch = batch
        self.latent = latent

        # Data
        self.dataset = dataset
        self.batch = batch
        self.train_data, self.valid_data = data(self.dataset,
                                                self.batch)

        # Training parameters
        self.epochs = epochs
        self.epochsrun = 0

        # Logging
        self.logfreq = logfreq
        self.plotfreq = plotfreq

        # Initialise DDP
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.encoder = DDP(self.encoder, device_ids=[self.device])
            self.decoder = DDP(self.decoder, device_ids=[self.device])
            self.discriminator = DDP(self.discriminator, device_ids=[self.device])

    @staticmethod
    def setup(seed=42):
        """Setup the training process"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            torch.mps.empty_cache()
        if os.getenv("DEBUG_MODE", 'False').lower() in ('true', '1', 't'):
            torch.autograd.set_detect_anomaly(True)

    def reconstruction(self, images, train=True):
        """Performs the reconstruction training step, in which the autoencoder updates
        the encoder and the decoder to minimize the reconstruction error

        Parameters
        ----------
        images : torch.tensor
            Flow's history

        train : bool, default=True
            Training mode

        Returns
        -------
            Reconstruction Loss

        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        self.discriminator.eval()

        if train:
            self.encoder_optimiser.zero_grad(set_to_none=True)
            self.decoder_optimiser.zero_grad(set_to_none=True)

        z = self.encoder(images)
        recon = self.decoder(z)
        loss = self.autoencoder_criterion()(recon, images)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.encoder_optimiser.step()
            self.decoder_optimiser.step()

        if train:
            return loss
        return images, recon, loss

    def regularisation(self, images, train=True):
        """Performs the regularisation training step, in which the adversarial network updates
        its discriminative network then updates its generator (which is also the
        encoder of the autoencoder)

        Parameters
        ----------
        images : torch.tensor
            Flow's history

        train : bool, default=True
            Training mode

        Returns
        -------
            Regularisations' Loss

        """
        if train:
            self.discriminator.train()
            self.discriminator_optimiser.zero_grad(set_to_none=True)
        else:
            self.discriminator.eval()
        self.encoder.eval()
        self.decoder.eval()

        # Train discriminator on generated data
        zfake = self.encoder(images)
        pred = self.discriminator(zfake)
        fake = torch.ones(pred.shape, device=self.device)
        lossfake = self.adversarial_criterion()(pred, fake)
        if train:
            lossfake.backward()

        # Train discriminator on true data
        zreal = torch.randn(*zfake.shape, device=self.device)
        pred = self.discriminator(zreal)
        real = torch.zeros(pred.shape, device=self.device)
        lossreal = self.adversarial_criterion()(pred, real)
        if train:
            lossreal.backward()
        Dloss = lossfake + lossreal

        # Discriminator update
        if train:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.discriminator_optimiser.step()

        # Train the generator
        if train:
            self.encoder.train()
            self.discriminator.eval()
            self.generator_optimiser.zero_grad(set_to_none=True)

        zfake = self.encoder(images)
        pred = self.discriminator(zfake)
        real = torch.zeros(pred.shape, device=self.device)
        Gloss = self.adversarial_criterion()(pred, real)

        if train:
            Gloss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.generator_optimiser.step()

        return Dloss, Gloss

    def train(self, data):
        """Training step.

        Parameters
        ----------
        data : DataLoader
            Training dataset

        epoch : int
            Training epoch

        Returns
        -------
            Training Loss

        """
        totalAE, totalD, totalG = 0., 0., 0.

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        for images, _ in data:
            images = images.to(self.device)
            AEloss = self.reconstruction(images)
            Dloss, Gloss = self.regularisation(images)
            totalAE += AEloss.item() * images.shape[0]
            totalD += Dloss.item()
            totalG += Gloss.item()
        return (totalAE / len(data.dataset),
                totalD / len(data.dataset),
                totalG / len(data.dataset))

    def validate(self, data, epoch):
        """Validation step.

        Parameters
        ----------
        data : DataLoader
            Validation dataset

        epoch : int
            Training epoch

        Returns
        -------
            Validation Loss

        """
        total, log = 0., {}
        with torch.no_grad():
            for images, labels in data:
                images = images.to(self.device)
                images, recon, AEloss = self.reconstruction(images, train=False)
                total += AEloss.item() * images.shape[0]
                if epoch % self.logfreq == 0 and not log:
                    log = self.select(recon, images, labels, log)
        return total / len(data.dataset), log

    def logging(self, epoch, log):
        """Log training to wandb and to console.

        Parameters
        ----------
        log : dict
            Dictionary containing data to be logged

        """
        if self.plotfreq != -1 and (epoch % self.plotfreq == 0 or epoch == self.epochs - 1):
            fig = self.plot(log)
            wandb.log({"Training Loss": log.get("Training Loss")[-1],
                       "Discriminator Loss": log.get("Discriminator Loss")[-1],
                       "Generator Loss": log.get("Generator Loss")[-1],
                       "Validation Loss": log.get("Validation Loss")[-1],
                       "Validation Results": fig},
                      step=log.get("Epochs")[-1])
            plt.close(fig)
        else:
            wandb.log({"Training Loss": log.get("Training Loss")[-1],
                       "Discriminator Loss": log.get("Discriminator Loss")[-1],
                       "Generator Loss": log.get("Generator Loss")[-1],
                       "Validation Loss": log.get("Validation Loss")[-1]},
                      step=log.get("Epochs")[-1])

    def plot(self, log):
        """Visualise the latent space and results. 

        Paramters
        ---------
        log : dict
            Dictionary containing log information

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure with the visualisation

        """
        matplotlib.rcParams.update({'font.size': 8})
        fig = plt.figure(figsize=(12, 2))
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.,
                                  width_ratios=[0.3, 0.7])

        for i in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(1 if i == 0 else 2,
                                                     1 if i == 0 else 10,
                                                     subplot_spec=outer[i],
                                                     wspace=0.2, hspace=0)
            # Visualise the latent space
            if i == 0:
                ax = plt.Subplot(fig, inner[0])
                for image, label in self.valid_data:
                    z = self.encoder(image.to(self.device)).to('cpu').detach().numpy()
                    im = ax.scatter(z[:, 0], z[:, 1], s=3.0, c=label, cmap='tab10', alpha=.65)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title("Latent Space")
                fig.add_subplot(ax)

            # Visualise the results
            if i == 1:
                for j, _ in enumerate(log["Temporal"]["labels"]):
                    ax = plt.Subplot(fig, inner[j])
                    im = ax.imshow(log["Temporal"]["images"][j].cpu().squeeze())
                    ax.set_yticks([])
                    ax.set_xticks([])
                    if j == 0:
                        ax.set_ylabel("Original")
                    fig.add_subplot(ax)

                    ax = plt.Subplot(fig, inner[j + len(log["Temporal"]["labels"])])
                    im = ax.imshow(log["Temporal"]["recon"][j].cpu().squeeze())
                    ax.set_yticks([])
                    ax.set_xticks([])
                    if j == 0:
                        ax.set_ylabel("Recon")
                    fig.add_subplot(ax)

        fig.get_axes()[0].remove()
        return fig

    def select(self, recon, images, labels, log):
        """Select the individuals on validation set in which
        best fit an equidistant temporal mesh.

        Parameters
        ----------
        images : torch.tensor
            True data

        log : dict
            Dictionary containing log information

        recon : torch.tensor
            Reconstructed data

        labels : torch.tensor
            Labels

        Returns
        -------
            Dictionary containing the updated log information

        """
        empty = torch.tensor([], requires_grad=False, device=self.device)

        # Select the 1st object for each class
        indices = torch.tensor([], dtype=torch.int, requires_grad=False, device=self.device)
        for target in labels.unique():
            indices = torch.cat((indices,
                                 (labels == target).nonzero(as_tuple=True)[0][0].reshape(1).to(self.device)),
                                0)

        # Calculate the new losses
        loss = torch.mean(self.autoencoder_criterion(reduction='none')(recon, images),
                          dim=[1, 2, 3]).squeeze()

        # Update the log
        log.update(
            {"Temporal": {"loss": torch.cat([log.get("Temporal", {}).get("loss", empty), loss])[indices],
                          "images": torch.cat([log.get("Temporal", {}).get("images", empty), images])[indices],
                          "recon": torch.cat([log.get("Temporal", {}).get("recon", empty), recon])[indices],
                          "labels": labels.to(self.device)[indices]}}
        )
        return log

    def run(self):
        """Returns a trained model"""

        # Setup the training
        self.setup()

        # Set the logging
        log = {"Epochs": [], "Training Loss": [],
               "Discriminator Loss": [], "Generator Loss": [],
               "Validation Loss": []}

        # Train the model
        with progressbar.ProgressBar(min_value=self.epochsrun, max_value=self.epochs) as bar:
            for epoch in range(self.epochsrun, self.epochs):

                # Logging
                bar.update(epoch + 1)
                log['Epochs'].append(epoch)

                # Training step
                AEloss, Dloss, Gloss = self.train(self.train_data)
                log['Training Loss'].append(AEloss)
                log['Discriminator Loss'].append(Dloss)
                log['Generator Loss'].append(Gloss)

                # Validation step
                valid_loss, valid_log = self.validate(self.valid_data, epoch)
                log['Validation Loss'].append(valid_loss)

                # Logging
                if epoch % self.logfreq == 0 or epoch == self.epochs - 1:
                    log.update(valid_log)
                    self.logging(epoch, log)

        return self.encoder, self.decoder, self.discriminator
