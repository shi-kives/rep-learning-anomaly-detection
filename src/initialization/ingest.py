import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

paths = []

train_dataset = datasets.ImageFolder(root='datasets/MVTecAD/wood/train', transform = data_transform)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = True
)

iter = 1
for images, labels in train_loader:
    print(f"loading batch {iter}")
    iter += 1

print("-- data loaded successfully --")