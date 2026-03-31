# Fashion LDM (Latent Diffusion Model) on Modal

This project implements a Latent Diffusion Model (LDM) for generating fashion-related images, optimized to run on **Modal** for serverless GPU compute.

## Project Structure

- `train_modal.py`: The main entry point for running training, sampling, and the Streamlit app on Modal.
- `streamlit_app.py`: A Streamlit interface for interactive image generation.
- `config.py`: Configuration settings for model architecture, training hyperparameters, and paths.
- `src/`: Core source code.
  - `models/`: VAE and UNet model definitions.
  - `training/`: Training loops for VAE and Diffusion models.
  - `data/`: Data loading and preprocessing utilities.

## Prerequisites

1.  **Modal Account**: Sign up at [modal.com](https://modal.com).
2.  **Modal Client**: Install the Modal client locally:
    ```bash
    pip install modal
    ```
3.  **Authentication**: Set up your Modal token:
    ```bash
    modal token set --token-id <your-id> --token-secret <your-secret>
    ```

## Usage

### 1. Run Training
To start the full training pipeline (VAE training followed by Diffusion training) on a remote GPU:
```bash
modal run train_modal.py::main
```

### 2. Run Streamlit App
Once your model is trained, you can launch an interactive Streamlit web app to generate images:
```bash
modal serve train_modal.py
```
After running this command, Modal will provide a URL where you can access the app in your browser.

### 3. Run Specific Tasks
You can also run individual components of the pipeline:

- **Train VAE only**:
  ```bash
  modal run train_modal.py::train_vae_remote
  ```
- **Train Diffusion only**:
  ```bash
  modal run train_modal.py::train_diffusion_remote
  ```
- **Generate Samples (CLI)**:
  ```bash
  modal run train_modal.py::sample_remote
  ```

### 4. Download Results
After training or sampling, you can download the checkpoints and generated images from the Modal Volume to your local `./outputs` directory:
```bash
modal run train_modal.py::download_outputs
```

## Configuration

Most settings can be adjusted in `config.py`. When running on Modal, `train_modal.py` automatically overrides path-related settings to ensure they point to the persistent Modal Volume mounted at `/mnt/fashion-ldm`.

## Notes

- **GPU**: The project is configured to use an NVIDIA H100 GPU on Modal.
- **Persistence**: All training progress and checkpoints are saved to a Modal Volume named `fashion-ldm-vol`, ensuring data persists across different runs.

## 👥 Team Members

This project was developed by:

*   SeifElden Osama
*   Sama NigmEldin
*   Habiba Ashraf
*   Mohamed Badr
*   Mohamed AbdAlwanis
