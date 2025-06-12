import streamlit as st
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import urllib.request
from urllib.error import URLError
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.augmentation import TrainAugmentation, TestAugmentation
from models.patches import Patches, PatchEncoder
from models.encoder import Encoder
from models.decoder import Decoder
from models.mae import MaskedAutoencoder
from models.utils import calculate_accuracy, calculate_psnr, calculate_ssim, save_model, load_model
from config import *

def main():
    st.set_page_config(layout="wide")
    st.title("Masked Autoencoder (MAE) with PyTorch")

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")

        batch_size = st.slider("Batch Size", 32, 512, BATCH_SIZE, step=32) 
        epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=1000, value=10)

        # Scheduler parameters
        st.subheader("Learning Rate Scheduler")
        use_scheduler = st.checkbox("Use ReduceLROnPlateau Scheduler", value=True)
        scheduler_factor = st.slider("Reduction Factor", 0.1, 0.9, 0.1, step=0.05) if use_scheduler else 0.1
        patience = st.slider("Patience (epochs)", 1, 10, 3) if use_scheduler else 3
        min_lr = st.number_input("Minimum Learning Rate", 1e-6, 1e-3, 1e-5) if use_scheduler else 1e-5
               
        # Mask proportion
        mask_proportion = st.slider("Masking Proportion", 
                                  min_value=0.1, 
                                  max_value=0.9, 
                                  value=MASK_PROPORTION, 
                                  step=0.05,
                                  help="Proportion of patches to mask")

        # Visualization controls
        st.header("Sample Visualization")
        selected_batch = st.number_input("Batch Number", min_value=0, value=0,
                                       help="Select which batch to visualize")
        sample_range = st.slider("Sample Range", 
                                min_value=1, 
                                max_value=100, 
                                value=(1, 5),
                                help="Range of samples to display from selected batch")

        if st.button("Train Model"):
            train_model = True
        else:
            train_model = False

        st.write("")
        st.divider()
        
        # Model management
        st.header("Model Management")
        model_filename = st.text_input("Save Model filename", "mae_model.pkl")

        if st.button("Load Model"):
            load_model_flag = True
        else:
            load_model_flag = False

        st.divider()

        # Custom image
        st.header("Custom Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        st.divider()

    # Data Loading
    @st.cache_data
    def load_data():
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                urllib.request.urlopen('https://www.google.com', timeout=5)
                train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
                return train_dataset, test_dataset
                
            except (URLError, Exception) as e:
                retry_count += 1
                st.warning(f"Download attempt {retry_count} failed: {str(e)}")
                if retry_count == max_retries:
                    st.error("Failed to download dataset after multiple attempts. Please check your internet connection.")
                    st.stop()
                time.sleep(2)

    try:
        train_dataset, test_dataset = load_data()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_augmentation = TrainAugmentation().to(device)
    test_augmentation = TestAugmentation().to(device)
    patch_layer = Patches().to(device)
    patch_encoder = PatchEncoder(mask_proportion=mask_proportion).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    mae_model = MaskedAutoencoder(
        train_augmentation, test_augmentation,
        patch_layer, patch_encoder, encoder, decoder
    ).to(device)

    optimizer = torch.optim.Adam(mae_model.parameters())

    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 
                                    mode='min',
                                    factor=scheduler_factor,
                                    patience=patience,
                                    min_lr=min_lr,
                                    verbose=True)
    
    # Track best metrics
    best_loss = float('inf')
    best_accuracy = 0.0
    best_psnr = 0.0
    best_ssim = 0.0
    best_model_state = None    

    # Load model if requested
    if load_model_flag:
        loaded_model = load_model(model_filename)
        if loaded_model:
            mae_model = loaded_model.to(device)
            st.success(f"Model loaded successfully from {model_filename}!")
        else:
            st.error(f"Failed to load model from {model_filename}")

    # Function to display samples from a specific batch and range
    def display_batch_samples(loader, batch_idx, sample_start, sample_end):
        for i, (images, _) in enumerate(loader):
            if i == batch_idx:
                selected_batch = images
                break
        
        st.subheader(f"Samples {sample_start}-{sample_end} from Batch {batch_idx}")
        num_samples = sample_end - sample_start + 1
        cols = st.columns(num_samples)
        
        for i in range(sample_start-1, sample_end):
            if i >= len(selected_batch):
                break
            with cols[i - (sample_start-1)]:
                img = selected_batch[i].permute(1, 2, 0).numpy()
                st.image(img, caption=f"Sample {i+1}", use_container_width=True)

    # Process custom uploaded image
    if uploaded_file is not None:
        st.subheader("Custom Image Processing")
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            if st.button("Process Custom Image"):
                with torch.no_grad():
                    mae_model.eval()
                    (_, _, _, _, augmented_image,
                     mask_indices, unmask_indices, reconstructed_image) = mae_model.calculate_loss(image_tensor, test=True)
                    
                    # Get the patches
                    patches = patch_layer(augmented_image)
                    masked_patch, _ = patch_encoder.generate_masked_image(patches, unmask_indices)
                    masked_image = patch_layer.reconstruct_from_patch(masked_patch).cpu().numpy()
                    
                    # Calculate metrics
                    accuracy = calculate_accuracy(augmented_image, reconstructed_image)
                    psnr = calculate_psnr(augmented_image, reconstructed_image)
                    ssim = calculate_ssim(augmented_image, reconstructed_image)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(augmented_image[0].cpu().permute(1, 2, 0).numpy(), 
                               caption="Processed Image", use_container_width=True)
                    with col2:
                        st.image(masked_image, 
                               caption=f"Masked Image ({mask_proportion*100:.0f}% masked)", 
                               use_container_width=True)
                    with col3:
                        st.image(reconstructed_image[0].cpu().permute(1, 2, 0).numpy(), 
                               caption=f"Reconstructed Image\nAccuracy: {accuracy:.2%}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}", 
                               use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Display samples from selected batch
    if not train_model and not load_model_flag:
        st.subheader("Data Visualization")
        if st.button("Show Selected Batch Samples"):
            display_batch_samples(train_loader, selected_batch, sample_range[0], sample_range[1])
        else:
            sample_images, _ = next(iter(train_loader))
            sample_images = sample_images[:5]

            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    img = sample_images[i].detach().cpu().permute(1, 2, 0).numpy()
                    st.image(img, caption=f"Sample {i+1}", use_container_width=True)

            st.write("Adjust the parameters in the sidebar and click 'Train Model' to start training.")

    # Training and Visualization
    if train_model:
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_history = []
        mae_history = []
        accuracy_history = []
        psnr_history = []
        ssim_history = []
        lr_history = []

        # Create columns for charts
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("**Training Loss**")
            loss_chart = st.line_chart()
        with col2:
            st.markdown("**Mean Absolute Error (MAE)**")
            mae_chart = st.line_chart()
        with col3:
            st.markdown("**Pixel Accuracy**")
            accuracy_chart = st.line_chart()
        with col4:
            st.markdown("**Peak SNR (PSNR)**")
            psnr_chart = st.line_chart()
        with col5:
            st.markdown("**Structural Similarity (SSIM)**")
            ssim_chart = st.line_chart()

        results_container = st.container()

        for epoch in range(epochs):
            mae_model.train()
            total_loss = 0
            total_mae = 0
            total_accuracy = 0
            total_psnr = 0
            total_ssim = 0

            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(device)
                optimizer.zero_grad()
                loss, mae, _, _, augmented_images, _, _, reconstructed_images = mae_model.calculate_loss(images)
                
                # Calculate metrics
                accuracy = calculate_accuracy(augmented_images, reconstructed_images)
                psnr = calculate_psnr(augmented_images, reconstructed_images)
                ssim = calculate_ssim(augmented_images, reconstructed_images)
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_mae += mae.item()
                total_accuracy += accuracy
                total_psnr += psnr
                total_ssim += ssim

                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    status_text.text(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}\n"
                                   f"Loss: {loss.item():.4f}, MAE: {mae.item():.4f}, Accuracy: {accuracy:.2%}\n"
                                   f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}\n"
                                   f"LR: {current_lr:.2e}")
                    progress_bar.progress((epoch * len(train_loader) + batch_idx + 1) / (epochs * len(train_loader)))

            avg_loss = total_loss / len(train_loader)
            avg_mae = total_mae / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)
            avg_psnr = total_psnr / len(train_loader)
            avg_ssim = total_ssim / len(train_loader)

            # Update learning rate scheduler
            if use_scheduler:
                scheduler.step(avg_loss)

            # Track best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = mae_model.state_dict()
                
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                
            # Track current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            loss_history.append(avg_loss)
            mae_history.append(avg_mae)
            accuracy_history.append(avg_accuracy)
            psnr_history.append(avg_psnr)
            ssim_history.append(avg_ssim)
            lr_history.append(current_lr)

            # Update charts
            loss_chart.add_rows({"Training Loss": [avg_loss]})
            mae_chart.add_rows({"MAE": [avg_mae]})
            accuracy_chart.add_rows({"Accuracy": [avg_accuracy]})
            psnr_chart.add_rows({"PSNR": [avg_psnr]})
            ssim_chart.add_rows({"SSIM": [avg_ssim]})

            if (epoch + 1) % 5 == 0 or epoch == 0:
                mae_model.eval()
                with torch.no_grad():
                    # Get the selected batch for visualization
                    for i, (test_images, _) in enumerate(test_loader):
                        if i == selected_batch:
                            break
                    
                    # Get the requested range of samples
                    test_images = test_images[sample_range[0]-1:sample_range[1]].to(device)
                    (_, _, _, _, augmented_images,
                    mask_indices, unmask_indices, reconstructed_images) = mae_model.calculate_loss(test_images, test=True)
                    
                    # Calculate test metrics
                    test_accuracy = calculate_accuracy(augmented_images, reconstructed_images)
                    test_psnr = calculate_psnr(augmented_images, reconstructed_images)
                    test_ssim = calculate_ssim(augmented_images, reconstructed_images)

                    with results_container:
                        with st.expander(f"Epoch {epoch+1} Results (Batch {selected_batch}, Samples {sample_range[0]}-{sample_range[1]})", expanded=epoch==0):
                            st.subheader(f"Epoch {epoch+1} Results")
                            
                            # Add metrics to the expander
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Loss", f"{avg_loss:.4f}")
                                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                            with col2:
                                st.metric("Current MAE", f"{avg_mae:.4f}")
                                st.metric("Test PSNR", f"{test_psnr:.2f} dB")
                            with col3:
                                st.metric("Current Accuracy", f"{avg_accuracy:.2%}")
                                st.metric("Test SSIM", f"{test_ssim:.4f}")
                            with col4:
                                st.metric("Learning Rate", f"{current_lr:.2e}")
                            
                            # Visualization section
                            st.write("### Sample Visualizations")
                            num_samples = sample_range[1] - sample_range[0] + 1
                            cols = st.columns(num_samples)

                            for i in range(num_samples):
                                if i >= len(test_images):
                                    break
                                with cols[i]:
                                    # Original images
                                    orig_img = augmented_images[i].cpu().permute(1, 2, 0).numpy()
                                    st.image(orig_img, caption=f"Original {sample_range[0]+i}", use_container_width=True)

                                    # Masked images
                                    patches = patch_layer(augmented_images[i].unsqueeze(0))
                                    masked_patch, _ = patch_encoder.generate_masked_image(patches, unmask_indices[i].unsqueeze(0))
                                    masked_img = patch_layer.reconstruct_from_patch(masked_patch).cpu().numpy()
                                    st.image(masked_img, caption=f"Masked {sample_range[0]+i}", use_container_width=True)

                                    # Reconstructed images
                                    recon_img = reconstructed_images[i].cpu().permute(1, 2, 0).numpy()
                                    st.image(recon_img, 
                                           caption=f"Reconstructed {sample_range[0]+i}\nPSNR: {test_psnr:.2f} dB\nSSIM: {test_ssim:.4f}", 
                                           use_container_width=True)

                mae_model.train()

        # Display best metrics
        st.subheader("Training Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Loss", f"{best_loss:.4f}")
        with col2:
            st.metric("Best Accuracy", f"{best_accuracy:.2%}")
        with col3:
            st.metric("Best PSNR", f"{best_psnr:.2f} dB")
        with col4:
            st.metric("Best SSIM", f"{best_ssim:.4f}")
        
        # Load best model state
        if best_model_state is not None:
            mae_model.load_state_dict(best_model_state)
            st.info("Loaded best model weights based on validation loss.")
        
        #Save the trained model
        saved_filename = save_model(mae_model, model_filename)
        st.success(f"Training completed! Model saved to {saved_filename}")

if __name__ == '__main__':
    main()