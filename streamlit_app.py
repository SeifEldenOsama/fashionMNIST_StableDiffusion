import streamlit as st
import torch
import os
import sys
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, ".")

from config import config
from src.training.sampler import sample_latent_images

# Configuration for Streamlit
st.set_page_config(page_title="Fashion LDM Generator", layout="wide")

st.title("👗 Fashion LDM Image Generator")
st.markdown("Generate fashion images using your trained Latent Diffusion Model.")

# Sidebar for parameters
st.sidebar.header("Generation Parameters")
n_samples = st.sidebar.slider("Number of samples", 1, 16, 4)
guidance_scale = st.sidebar.slider("Guidance Scale", 0.0, 10.0, 2.0, 0.5)
seed = st.sidebar.number_input("Random Seed", value=42)

# Load model and generate
if st.button("Generate Images"):
    with st.spinner("Generating fashion images..."):
        try:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Update config with UI parameters
            config.N_SAMPLES = n_samples
            config.GUIDANCE_WEIGHT = guidance_scale
            
            # Set seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # Call the sampling function
            # Note: We assume the model paths are already correctly set in config by the Modal wrapper
            from src.training.sampler import sample_latent_images
            
            # We'll use a temporary directory for the UI samples
            temp_samples_dir = "./ui_samples"
            os.makedirs(temp_samples_dir, exist_ok=True)
            config.SAMPLES_DIR = temp_samples_dir
            
            # Run sampling
            sample_latent_images(device, suffix="streamlit")
            
            # Display results
            st.subheader("Generated Results")
            
            # Find the latest generated grid or images
            sample_files = [f for f in os.listdir(temp_samples_dir) if f.endswith(".png")]
            if sample_files:
                # Sort by modification time to get the latest
                sample_files.sort(key=lambda x: os.path.getmtime(os.path.join(temp_samples_dir, x)), reverse=True)
                
                # Display the first (latest) image
                latest_image_path = os.path.join(temp_samples_dir, sample_files[0])
                img = Image.open(latest_image_path)
                st.image(img, caption=f"Generated Fashion Images (Seed: {seed})", use_container_width=True)
            else:
                st.error("No images were generated. Check if the model checkpoints exist.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Make sure you have trained the model and the checkpoints are available in the volume.")

st.sidebar.markdown("---")
st.sidebar.info("This app runs on Modal using a serverless GPU.")
