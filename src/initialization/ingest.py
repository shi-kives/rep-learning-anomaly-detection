import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

class ImageDataset(Dataset):
    def __init__(self, dir, transform = transform):
        self.dir = dir
        self.images = os.listdir(dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dir, self.images[idx])
        image = np.array(Image.open(image_path))

        if self.transform:
            image = self.transform(image)

        return image

path = 'datasets/MVTecAD/wood/train/good'

dataset = ImageDataset(path)
dataset_length = len(dataset)

print("number of training examples: ", dataset_length)

data_loader = DataLoader(
    dataset = dataset,
    batch_size = 32,
    shuffle = True
)

print("number of batches: ", len(data_loader))
print("data loaded successfully.")