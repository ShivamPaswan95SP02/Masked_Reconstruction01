import torch
import torch.nn as nn
import torch.nn.functional as F

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

        return loss, mae, loss_patch, loss_output, augmented_images, mask_indices, unmask_indices, decoder_outputs

    def forward(self, x):
        return self.calculate_loss(x)