"""Utility methods for model training and evaluation"""

# pylint: disable=locally-disabled, redefined-outer-name, not-callable, invalid-name, too-many-arguments, unused-argument, invalid-unary-operand-type, arguments-differ, super-init-not-called, arguments-renamed, import-error

import os
import random
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import progressbar
import torch
import wandb
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from utils.dataset import data

PIL.Image.MAX_IMAGE_PIXELS = None


__all__ = ["AETrainer", "disabled_train"]


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
        self.device = int(os.environ["LOCAL_RANK"])

        # Dataset
        self.dataset = dataset
        self.batch = batch
        self.train_data, self.valid_data = data(self.dataset,
                                                self.batch)

        # Model ojects
        self.model = model.to(self.device)
        self.architecture = type(model).__name__

        # Optimiser
        self.optimiser = Adam(model.parameters(), lr=lr, betas=betas)

        # Loss
        self.criterion = MSELoss

        # Training
        self.epochs = epochs
        self.epochsrun = 0

        # Logging
        self.logfreq = logfreq
        self.plotfreq = plotfreq
        directory = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(os.path.dirname(directory), "experiments", dataset)
        os.makedirs(os.path.join(folder, type(model).__name__), exist_ok=True)
        self.savepath = os.path.join(os.path.dirname(directory),
                                     "experiments", dataset, type(model).__name__)

        # Initialise DDP
        self.model = DDP(self.model, device_ids=[self.device])

    @abstractmethod
    def train(self, data, epoch):
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
                train_loss = self.train(self.train_data, epoch)
                log['Training Loss'].append(train_loss)

                # Validation step
                valid_loss, valid_log = self.validate(self.valid_data, epoch)
                log['Validation Loss'].append(valid_loss)

                # Logging
                if epoch % self.logfreq == 0 or epoch == self.epochs - 1:
                    log.update(valid_log)
                    self.logging(epoch, log)

        # Save the model
        torch.distributed.barrier()
        if self.device == int(os.environ["MAIN_RANK"]):
            self.save(epoch)

        return self.model

    def save(self, epoch):
        """Save the current snapshot of the training model.

        Parameters
        ----------
        epochs : int
            Training epochs

        """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
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
        if self.plotfreq != -1 and epoch % self.plotfreq == 0:
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
        """Visualise the prediction throughout the temporal domain

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
        fig, ax = plt.subplots(2, len(log["Temporal"]["labels"]), figsize=(8, 2))
        for j, _ in enumerate(log["Temporal"]["labels"]):
            ax[0, j].imshow(log["Temporal"]["images"][j].cpu().squeeze())
            ax[0, j].set_yticks([])
            ax[0, j].set_xticks([])
            loss = log["Temporal"]["loss"][j]
            ax[0, j].set_title(f"Pointwise Error\n(Loss: {loss:.4e})")

            ax[1, j].imshow(log["Temporal"]["recon"][j].cpu().squeeze())
            ax[1, j].set_yticks([])
            ax[1, j].set_xticks([])

            if j == 0:
                ax[0, j].set_ylabel("Original")
                ax[1, j].set_ylabel("Recon")
        fig.tight_layout()
        return fig

    def select(self, recon, images, labels, log, n=8):
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

        n : int, default=8
            Number of elements to be stored for logging

        Returns
        -------
            Dictionary containing the updated log information

        """
        empty = torch.tensor([], requires_grad=False, device=self.device)

        # Select the 1st object for each class
        indices = torch.tensor([], dtype=torch.int, requires_grad=False, device=self.device)
        for target in labels.unique():
            indices = torch.cat((indices,
                                 (labels == target).nonzero(as_tuple=True)[0][0].reshape(1)),
                                0)

        # Calculate the new losses
        loss = torch.mean(self.criterion(reduction='none')(recon, images),
                          dim=[1, 2, 3]).squeeze()

        # Update the log
        log.update(
            {"Temporal": {"loss": torch.cat([log.get("Temporal", {}).get("loss", empty), loss])[indices],
                          "images": torch.cat([log.get("Temporal", {}).get("images", empty), images])[indices],
                          "recon": torch.cat([log.get("Temporal", {}).get("recon", empty), recon])[indices],
                          "labels": labels[indices]}}
        )

        return log


class AETrainer(BaseTrainer):
    """Class to perform the training of a model"""

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

    def train(self, data, epoch):
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
        self.train_data.sampler.set_epoch(epoch)
        scaler = torch.cuda.amp.GradScaler()
        for images, _ in data:
            self.optimiser.zero_grad(set_to_none=True)
            images = images.to(self.device)
            with torch.autocast(device_type='cuda'
                                if torch.cuda.is_available() else 'cpu',
                                dtype=torch.float16):
                recon = self.model(images)
                loss = self.criterion()(recon, images)
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimiser)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(self.optimiser)
            scaler.update()
            self.optimiser.zero_grad(set_to_none=True)
            total += loss.item() * images.size(0) / len(data.dataset)
            del loss
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
                with torch.autocast(device_type='cuda'
                                    if torch.cuda.is_available() else 'cpu',
                                    dtype=torch.float16):
                    recon = self.model(images)
                    loss = self.criterion()(recon, images)
                total += (loss.item() * images.size(0) /
                          len(data.dataset))
                del loss
                if epoch % self.logfreq == 0:
                    log = self.select(recon, images, labels, log)
        return total, log


def disabled_train(self):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore"""
    return self
