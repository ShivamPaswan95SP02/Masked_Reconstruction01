import torch.nn as nn
from torchvision import transforms

class TrainAugmentation(nn.Module):
    def __init__(self, image_size=48):
        super().__init__()
        self.resize = transforms.Resize((32 + 20, 32 + 20))
        self.random_crop = transforms.RandomCrop((image_size, image_size))
        self.random_flip = transforms.RandomHorizontalFlip()

    def forward(self, x):
        x = self.resize(x)
        x = self.random_crop(x)
        x = self.random_flip(x)
        return x

class TestAugmentation(nn.Module):
    def __init__(self, image_size=48):
        super().__init__()
        self.resize = transforms.Resize((image_size, image_size))

    def forward(self, x):
        x = self.resize(x)
        return x