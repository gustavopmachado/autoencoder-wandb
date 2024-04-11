"""Methods to generate and visualise the data for training"""

# pylint: disable=import-error

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# pylint: disable=unsubscriptable-object

__all__ = ["data", "visualise"]


def data(dataset, batch, output="loader"):
    """Generate the data for training.

    Parameters
    ----------
    dataset : str
        Name of the dataset to be used, which will be downloaded to data/

    batch : int
        Batch size

    output : str, default="loader"
        Controls the output. If 'loader' only outputs the training and validation
        data loaders. Otherwise, also outputs the training and validation datasets

    Returns
    -------
    train : torch.utils.data.dataloader.DataLoader
        Train DataLoader

    valid : torch.utils.data.dataloader.DataLoader
        Validation DataLoader

    train_dataset : torch.utils.data.Dataset, optional
        Train Dataset

    valid_dataset : torch.utils.data.Dataset, optional
        Validation Dataset

    """
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path(f"./experiments/{dataset}").mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(),])

    if dataset == "MNIST":
        train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                       transform=transform)
        valid_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                       transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=True)

    if output == "loader":
        return train_loader, valid_loader
    return (train_loader, valid_loader), (train_dataset, valid_dataset)


def visualise(dataloader):
    """Visualise the DataLoader images in a 2x2 grid.

    Parameters
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        DataLoader to be visualised

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure with the visualisation

    """
    batch = next(iter(dataloader))
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat):
        ax.imshow(batch[0][i].squeeze(), alpha=.75)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig
