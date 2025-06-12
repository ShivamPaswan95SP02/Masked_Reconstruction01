import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from einops import rearrange
import time
import urllib.request
from urllib.error import URLError
import pickle
import os
from PIL import Image

from torch.optim.lr_scheduler import ReduceLROnPlateau
# Hyperparameters
BUFFER_SIZE = 1024
BATCH_SIZE = 256
IMAGE_SIZE = 48
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0.75
ENC_PROJECTION_DIM = 128
DEC_PROJECTION_DIM = 64
ENC_NUM_HEADS = 4
ENC_LAYERS = 6
DEC_NUM_HEADS = 4
DEC_LAYERS = 2
ENC_TRANSFORMER_UNITS = [ENC_PROJECTION_DIM * 2, ENC_PROJECTION_DIM]
DEC_TRANSFORMER_UNITS = [DEC_PROJECTION_DIM * 2, DEC_PROJECTION_DIM]
EPOCHS = 50
LAYER_NORM_EPS = 1e-6
PIXEL_ACCURACY_THRESHOLD = 0.1  # Threshold for considering a pixel "correct"

# Data Augmentation
class TrainAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = transforms.Resize((32 + 20, 32 + 20))
        self.random_crop = transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE))
        self.random_flip = transforms.RandomHorizontalFlip()

    def forward(self, x):
        x = self.resize(x)
        x = self.random_crop(x)
        x = self.random_flip(x)
        return x

class TestAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

    def forward(self, x):
        x = self.resize(x)
        return x

# Patches Layer
class Patches(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size = images.shape[0]
        patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=self.patch_size, p2=self.patch_size)
        return patches

    def reconstruct_from_patch(self, patch):
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = patch.reshape(num_patches, self.patch_size, self.patch_size, 3)
        rows = torch.chunk(patch, n, dim=0)
        rows = [torch.cat(torch.unbind(x, dim=0), dim=1) for x in rows]
        reconstructed = torch.cat(rows, dim=0)
        return reconstructed

# Patch Encoder
class PatchEncoder(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, projection_dim=ENC_PROJECTION_DIM,
               mask_proportion=MASK_PROPORTION, downstream=False):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        self.mask_token = nn.Parameter(torch.randn(1, patch_size * patch_size * 3))
        self.projection = nn.Linear(patch_size * patch_size * 3, projection_dim)
        self.position_embedding = nn.Embedding(NUM_PATCHES, projection_dim)

    def forward(self, patches):
        batch_size, num_patches, _ = patches.shape
        self.num_mask = int(self.mask_proportion * num_patches)

        positions = torch.arange(0, num_patches).unsqueeze(0).to(patches.device)
        pos_embeddings = self.position_embedding(positions)
        pos_embeddings = pos_embeddings.repeat(batch_size, 1, 1)

        patch_embeddings = self.projection(patches) + pos_embeddings

        if self.downstream:
            return patch_embeddings
        else:
            rand_indices = torch.argsort(torch.rand(batch_size, num_patches, device=patches.device), dim=-1)
            mask_indices = rand_indices[:, :self.num_mask]
            unmask_indices = rand_indices[:, self.num_mask:]

            unmasked_embeddings = torch.gather(patch_embeddings, 1,
                                            unmask_indices.unsqueeze(-1).expand(-1, -1, self.projection_dim))
            unmasked_positions = torch.gather(pos_embeddings, 1,
                                           unmask_indices.unsqueeze(-1).expand(-1, -1, self.projection_dim))

            masked_positions = torch.gather(pos_embeddings, 1,
                                         mask_indices.unsqueeze(-1).expand(-1, -1, self.projection_dim))
            mask_tokens = self.mask_token.repeat(batch_size, self.num_mask, 1)
            masked_embeddings = self.projection(mask_tokens) + masked_positions

            return (unmasked_embeddings, masked_embeddings,
                  unmasked_positions, mask_indices, unmask_indices)

    def generate_masked_image(self, patches, unmask_indices):
        patch = patches[0]
        unmask_index = unmask_indices[0]
        new_patch = torch.zeros_like(patch)
        new_patch[unmask_index] = patch[unmask_index]
        return new_patch, 0

# Encoder
class Encoder(nn.Module):
    def __init__(self, num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(ENC_PROJECTION_DIM, eps=LAYER_NORM_EPS),
                nn.MultiheadAttention(ENC_PROJECTION_DIM, num_heads, dropout=0.1, batch_first=True),
                nn.LayerNorm(ENC_PROJECTION_DIM, eps=LAYER_NORM_EPS),
                nn.Sequential(
                    nn.Linear(ENC_PROJECTION_DIM, ENC_TRANSFORMER_UNITS[0]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(ENC_TRANSFORMER_UNITS[0], ENC_TRANSFORMER_UNITS[1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x1 = norm1(x)
            attn_output, _ = attn(x1, x1, x1)
            x2 = x + attn_output
            x3 = norm2(x2)
            x3 = ff(x3)
            x = x2 + x3
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE):
        super().__init__()
        self.proj = nn.Linear(ENC_PROJECTION_DIM, DEC_PROJECTION_DIM)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(DEC_PROJECTION_DIM, eps=LAYER_NORM_EPS),
                nn.MultiheadAttention(DEC_PROJECTION_DIM, num_heads, dropout=0.1, batch_first=True),
                nn.LayerNorm(DEC_PROJECTION_DIM, eps=LAYER_NORM_EPS),
                nn.Sequential(
                    nn.Linear(DEC_PROJECTION_DIM, DEC_TRANSFORMER_UNITS[0]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(DEC_TRANSFORMER_UNITS[0], DEC_TRANSFORMER_UNITS[1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            ]))

        self.norm = nn.LayerNorm(DEC_PROJECTION_DIM, eps=LAYER_NORM_EPS)
        self.head = nn.Sequential(
            nn.Linear(DEC_PROJECTION_DIM * NUM_PATCHES, image_size * image_size * 3),
            nn.Sigmoid()
        )
        self.image_size = image_size

    def forward(self, x):
        x = self.proj(x)

        for norm1, attn, norm2, ff in self.layers:
            x1 = norm1(x)
            attn_output, _ = attn(x1, x1, x1)
            x2 = x + attn_output
            x3 = norm2(x2)
            x3 = ff(x3)
            x = x2 + x3

        x = self.norm(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        x = x.view(-1, 3, self.image_size, self.image_size)
        return x

# MAE Model
class MaskedAutoencoder(nn.Module):
    def __init__(self, train_augmentation, test_augmentation, patch_layer, patch_encoder, encoder, decoder):
        super().__init__()
        self.train_augmentation = train_augmentation
        self.test_augmentation = test_augmentation
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def calculate_loss(self, images, test=False):
        if test:
            augmented_images = self.test_augmentation(images)
        else:
            augmented_images = self.train_augmentation(images)

        patches = self.patch_layer(augmented_images)
        (unmasked_embeddings, masked_embeddings,
         unmasked_positions, mask_indices, unmask_indices) = self.patch_encoder(patches)

        encoder_outputs = self.encoder(unmasked_embeddings)
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = torch.cat([encoder_outputs, masked_embeddings], dim=1)
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        loss_patch = torch.gather(patches, 1, mask_indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        loss_output = torch.gather(decoder_patches, 1, mask_indices.unsqueeze(-1).expand(-1, -1, decoder_patches.shape[-1]))

        loss = F.mse_loss(loss_patch, loss_output)

        mae = F.l1_loss(loss_patch, loss_output)
        #loss = mae.copy()

        return loss, mae, loss_patch, loss_output, augmented_images, mask_indices, unmask_indices, decoder_outputs

    def forward(self, x):
        return self.calculate_loss(x)

def calculate_accuracy(original, reconstructed, threshold=PIXEL_ACCURACY_THRESHOLD):
    """
    Calculate pixel accuracy between original and reconstructed images.
    A pixel is considered correct if the absolute difference is less than threshold.
    """
    diff = torch.abs(original - reconstructed)
    correct_pixels = (diff < threshold).float()
    accuracy = correct_pixels.mean()
    return accuracy.item()

def save_model(model, filename='mae_model.pkl'):
    """Save the trained model to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return filename

def load_model(filename='mae_model.pkl'):
    """Load a trained model from pickle file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    # Streamlit app configuration
    st.set_page_config(layout="wide")
    st.title("Masked Autoencoder (MAE) with PyTorch")

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")

        batch_size = st.slider("Batch Size", 32, 512, BATCH_SIZE, step=32) 
        epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=1000, value=10)

        # Add scheduler parameters
        st.subheader("Learning Rate Scheduler")
        use_scheduler = st.checkbox("Use ReduceLROnPlateau Scheduler", value=True)
        scheduler_factor = st.slider("Reduction Factor", 0.1, 0.9, 0.1, step=0.05) if use_scheduler else 0.1
        patience = st.slider("Patience (epochs)", 1, 10, 3) if use_scheduler else 3
        min_lr = st.number_input("Minimum Learning Rate", 1e-6, 1e-3, 1e-5) if use_scheduler else 1e-5
               
        
        # Add slider for mask proportion
        mask_proportion = st.slider("Masking Proportion", 
                                  min_value=0.1, 
                                  max_value=0.9, 
                                  value=MASK_PROPORTION, 
                                  step=0.05,
                                  help="Proportion of patches to mask")

        # Add controls for batch selection and sample display
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

        # Add some space before the divider
        st.write("")  # Empty space
        
        # Add horizontal divider
        st.divider()
        

        # Add model save/load options
        st.header("Model Management")
        model_filename = st.text_input("Model filename", "mae_model.pkl")

        if st.button("Load Model"):
            load_model_flag = True
        else:
            load_model_flag = False

        # Add option to upload custom image
        st.header("Custom Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        # Add divider after Custom Image section
        st.divider()

    # Data Loading with retry mechanism
    @st.cache_data
    def load_data():
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check internet connection first
                urllib.request.urlopen('https://www.google.com', timeout=5)
                
                # Try to download CIFAR-100
                train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
                return train_dataset, test_dataset
                
            except (URLError, Exception) as e:
                retry_count += 1
                st.warning(f"Download attempt {retry_count} failed: {str(e)}")
                if retry_count == max_retries:
                    st.error("Failed to download dataset after multiple attempts. Please check your internet connection.")
                    st.stop()
                time.sleep(2)  # Wait before retrying

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

    # Initialize the scheduler
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 
                                    mode='min',  # Monitor loss
                                    factor=scheduler_factor,
                                    patience=patience,
                                    min_lr=min_lr,
                                    verbose=True)
    
    # Track best metrics
    best_loss = float('inf')
    best_accuracy = 0.0
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
        # Get the selected batch
        for i, (images, _) in enumerate(loader):
            if i == batch_idx:
                selected_batch = images
                break
        
        # Display the requested range of samples
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
            
            # Convert to tensor and process
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
                    
                    # Calculate accuracy
                    accuracy = calculate_accuracy(augmented_image, reconstructed_image)
                    
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
                               caption=f"Reconstructed Image (Accuracy: {accuracy:.2%})", 
                               use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Display samples from selected batch
    if not train_model and not load_model_flag:
        st.subheader("Data Visualization")
        if st.button("Show Selected Batch Samples"):
            display_batch_samples(train_loader, selected_batch, sample_range[0], sample_range[1])
        else:
            # Default display of first 5 samples from first batch
            sample_images, _ = next(iter(train_loader))
            sample_images = sample_images[:5]

            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    img = sample_images[i].permute(1, 2, 0).numpy()
                    st.image(img, caption=f"Sample {i+1}", use_container_width=True)

            st.write("Adjust the parameters in the sidebar and click 'Train Model' to start training.")

    # Training and Visualization
    if train_model:
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_history = []
        mae_history = []
        accuracy_history = []
        lr_history = []

        # Create columns for charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Training Loss**")
            loss_chart = st.line_chart()
        with col2:
            st.markdown("**Mean Absolute Error (MAE)**")
            mae_chart = st.line_chart()
        with col3:
            st.markdown("**Pixel Accuracy**")
            accuracy_chart = st.line_chart()


        results_container = st.container()

        for epoch in range(epochs):
            mae_model.train()
            total_loss = 0
            total_mae = 0
            total_accuracy = 0

            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(device)
                optimizer.zero_grad()
                loss, mae, _, _, augmented_images, _, _, reconstructed_images = mae_model.calculate_loss(images)
                
                # Calculate accuracy
                accuracy = calculate_accuracy(augmented_images, reconstructed_images)
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_mae += mae.item()
                total_accuracy += accuracy

                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    status_text.text(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}\n"
                                   f"Loss: {loss.item():.4f}, MAE: {mae.item():.4f}, Accuracy: {accuracy:.2%}\n"
                                   f"LR: {current_lr:.2e}")
                    progress_bar.progress((epoch * len(train_loader) + batch_idx + 1) / (epochs * len(train_loader)))


            avg_loss = total_loss / len(train_loader)
            avg_mae = total_mae / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)

            # Update learning rate scheduler
            if use_scheduler:
                scheduler.step(avg_accuracy)

            # PEAK SNR , STRUCTURAL SIMILARITY INDEX ( SSIM ) torchmetrics
            
            # Track best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = mae_model.state_dict()
                
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                
            # Track current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            
            loss_history.append(avg_loss)
            mae_history.append(avg_mae)
            accuracy_history.append(avg_accuracy)
            lr_history.append(current_lr)

            # Update charts with labeled data
            loss_chart.add_rows({"Training Loss": [avg_loss]})
            mae_chart.add_rows({"MAE": [avg_mae]})
            accuracy_chart.add_rows({"Accuracy": [avg_accuracy]})

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
                    
                    # Calculate test accuracy
                    test_accuracy = calculate_accuracy(augmented_images, reconstructed_images)

                    with results_container:
                        # Create an expander for this epoch's results
                        with st.expander(f"Epoch {epoch+1} Results (Batch {selected_batch}, Samples {sample_range[0]}-{sample_range[1]})", expanded=epoch==0):
                            st.subheader(f"Epoch {epoch+1} Results")
                            
                            # Add metrics to the expander
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Loss", f"{avg_loss:.4f}")
                            with col2:
                                st.metric("Current MAE", f"{avg_mae:.4f}")
                            with col3:
                                st.metric("Current Accuracy", f"{avg_accuracy:.2%}")
                                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                            
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
                                    st.image(recon_img, caption=f"Reconstructed {sample_range[0]+i}", use_container_width=True)

                            # Add some space between samples
                            st.write("")

                mae_model.train()


        # Display best metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Loss", f"{best_loss:.4f}")
        with col2:
            st.metric("Best Accuracy", f"{best_accuracy:.2%}")
        
        # Load best model state
        if best_model_state is not None:
            mae_model.load_state_dict(best_model_state)
            st.info("Loaded best model weights based on validation loss.")
        
        # Save the trained model
        #saved_filename = save_model(mae_model, model_filename)
        #st.success(f"Training completed! Model saved to {saved_filename}")

        # Save the trained model using pickle
        if st.button("Save Model"):
            saved_filename = save_model(mae_model, model_filename)  # Using the function you defined
            st.success(f"Model saved successfully to {saved_filename}!")

if __name__ == '__main__':
    main()