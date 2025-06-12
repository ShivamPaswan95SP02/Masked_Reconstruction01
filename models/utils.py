import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pickle
import os

def calculate_accuracy(original, reconstructed, threshold=0.1):
    diff = torch.abs(original - reconstructed)
    correct_pixels = (diff < threshold).float()
    accuracy = correct_pixels.mean()
    return accuracy.item()

def calculate_psnr(original, reconstructed):
    original_np = (original.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    reconstructed_np = (reconstructed.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    
    psnr_values = []
    for orig, recon in zip(original_np, reconstructed_np):
        psnr = peak_signal_noise_ratio(orig, recon, data_range=255)
        psnr_values.append(psnr)
    
    return np.mean(psnr_values)

def calculate_ssim(original, reconstructed):
    original_np = (original.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    reconstructed_np = (reconstructed.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    
    ssim_values = []
    for orig, recon in zip(original_np, reconstructed_np):
        ssim = structural_similarity(orig, recon, channel_axis=-1, data_range=255)
        ssim_values.append(ssim)
    
    return np.mean(ssim_values)

def save_model(model, filename='mae_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return filename

def load_model(filename='mae_model.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None