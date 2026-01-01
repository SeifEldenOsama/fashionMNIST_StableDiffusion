# FashionMNIST Stable Diffusion: Generative Modeling for Apparel

## Project Overview

This project implements a **Stable Diffusion** model tailored for the **FashionMNIST** dataset. The goal is to explore the capabilities of latent diffusion models in generating high-quality, novel images of apparel items, leveraging the foundational principles of modern generative AI. By adapting the diffusion architecture to the simpler, grayscale FashionMNIST domain, this work provides a clear, educational, and reproducible example of how these powerful models can be applied to specific image datasets.

The implementation is built from scratch using PyTorch, focusing on the core components of a latent diffusion model: the Variational Autoencoder (VAE) and the UNet denoiser.

## Key Features

*   **Latent Diffusion Architecture:** Full implementation of a latent diffusion model, including a custom VAE for compressing images into a lower-dimensional latent space and a UNet for the iterative denoising process.
*   **FashionMNIST Integration:** Specifically configured to train on the 28x28 grayscale images of the FashionMNIST dataset, demonstrating effective generative modeling on a structured, real-world image classification benchmark.
*   **PyTorch Implementation:** All models and training loops are implemented using the PyTorch framework, making the code highly flexible and suitable for further research and experimentation.
*   **Educational Notebooks:** Comprehensive Jupyter Notebooks guide the user through the entire process, from data loading and model definition to training and image generation.

## Technical Specifications

The project's architecture is composed of three main components: the VAE, the UNet, and the Diffusion Process.

### Model Architecture Summary

| Component | Role | Key Features |
| :--- | :--- | :--- |
| **Variational Autoencoder (VAE)** | Encodes images into a latent space and decodes latent vectors back to images. | Custom Encoder and Decoder modules, ChannelAttention, GroupNorm, SiLU activation. |
| **UNet** | The core denoising network. Predicts noise added to the latent vector at each diffusion step. | Includes CrossAttention and LatentResBlock components for robust feature extraction. |
| **Diffusion Process** | The iterative process of adding and removing noise to generate new samples. | Implements a 1000-step process with custom timestep embeddings. |

### Training Configuration

The following parameters were used for training the model, as detailed in the configuration class within the notebooks:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Dataset** | FashionMNIST | 28x28 grayscale images of 10 apparel classes. |
| **Image Size** | 28x28 | Standard resolution for the FashionMNIST dataset. |
| **Latent Size** | 7x7 | The spatial dimension of the compressed latent representation. |
| **Steps** | 1000 | The number of forward and reverse diffusion steps. |
| **Epochs** | 100 | Total number of training epochs. |
| **Batch Size** | 128 | Number of samples processed per training iteration. |
| **Learning Rate** | 1e-4 | The rate at which model weights are updated during training. |

## Repository Structure

The repository is organized to separate the model implementation, training scripts, and documentation:

```
.
├── model/                      # Directory for saved model checkpoints (.pt files).
├── notebooks/                  # Jupyter Notebooks for training and application.
│   ├── fashionMNIST_stableDiffusion.ipynb      # Main training and model definition notebook.
│   └── fashionMNIST_stableDiffusionApplication (1).ipynb # Notebook for image generation and application.
├── presentation/               # Project presentation or report materials.
└── README.md                   # This file.
```

## Getting Started

### Prerequisites

*   Python 3.x
*   A machine with a modern GPU (recommended for faster training).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SeifEldenOsama/fashionMNIST_StableDiffusion.git
    cd fashionMNIST_StableDiffusion
    ```

2.  **Install dependencies:**
    The project relies primarily on PyTorch and related libraries. A `requirements.txt` file is assumed to contain the following:
    ```bash
    pip install torch torchvision numpy matplotlib tqdm
    ```

### Usage

The primary way to interact with this project is through the Jupyter Notebooks in the `notebooks/` directory.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```

2.  **Training:** Open `fashionMNIST_stableDiffusion.ipynb` to execute the data loading, model definition, and training loop. This will save the trained VAE and UNet models to the `model/` directory.

3.  **Generation:** Open `fashionMNIST_stableDiffusionApplication (1).ipynb` to load the trained models and begin generating new, synthetic images of FashionMNIST items. You can experiment with different noise levels and sampling steps.

## Team Members

This project was developed by:

*   SeifElden Osama
*   Sama NigmEldin
*   Habiba Ashraf
*   Mohamed Badr
*   Mohamed AbdAlwanis

---
