# Autoencoders

This repository implements and compares **Autoencoders (AE)**, **Variational Autoencoders (VAE)**, and **Adversarial Autoencoders (AAE)**, focusing on how different regularisation strategies shape latent space geometry and training dynamics.

The work was developed to support a technical session on Deep Learning and MLOps, and later shared as accompanying material. The codebase is structured as a reproducible experimental framework rather than a standalone model implementation.

---

## Latent Space & Results

MNIST is used for its simplicity and because it supports stable training in this setting while remaining fully reproducible on local machines without GPU requirements.

The latent space is constrained to ℝ², making representation learning directly observable without post-hoc projections. Below is a snapshot of the latent space distribution and reconstruction behaviour across models.

**Autoencoder**  
![AE Latent Space](./images/AE%20Latent%20Space.png)

**Variational Autoencoder**  
![VAE Latent Space](./images/VAE%20Latent%20Space.png)

**Adversarial Autoencoder**  
![AAE Latent Space](./images/AAE%20Latent%20Space.png)

In this setup, the AE learns representations that prioritise reconstruction but remain globally unstructured, with noticeable class overlap. The VAE introduces a continuous latent structure aligned with a Gaussian prior, leading to smoother organisation but more diffuse class boundaries. The AAE, by matching the latent distribution adversarially, produces more compact and separated clusters, although with less explicit control over global distributional shape and increased sensitivity during training.

---

## Implementation & MLOps

Models are implemented from scratch in PyTorch, with minimal abstraction to retain full control over training behaviour. Training supports Distributed Data Parallel (DDP) where available.

Experiments are tracked using [Weights & Biases (W&B)](https://wandb.ai/home), with runs defined via external configuration. Metrics, latent projections, and reconstructions are logged as artifacts, and hyperparameter sweeps are used to explore sensitivity across optimisation and architectural choices. The figures below show an example sweep for the Adversarial Autoencoder.

![W&B Sweep Losses](./images/WandB%20Sweep%20Losses.png)
![W&B Sweep Hyperparameters](./images/WandB%20Sweep%20Hyperparameters.png)
![W&B Sweep Metrics](./images/WandB%20Sweep%20Metrics.png)

---

## Running

Dependencies can be installed via `pip` using the provided `requirements.txt`.

Training and hyperparameter sweeps are executed via the shell scripts under `scripts/`, with experiment configurations defined in `experiments/`. 