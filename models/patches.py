import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

class Patches(nn.Module):
    def __init__(self, patch_size=6):
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

class PatchEncoder(nn.Module):
    def __init__(self, patch_size=6, projection_dim=128,
               mask_proportion=0.75, downstream=False):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream
        self.num_patches = (48 // patch_size) ** 2  # Assuming image_size=48

        self.mask_token = nn.Parameter(torch.randn(1, patch_size * patch_size * 3))
        self.projection = nn.Linear(patch_size * patch_size * 3, projection_dim)
        self.position_embedding = nn.Embedding(self.num_patches, projection_dim)

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