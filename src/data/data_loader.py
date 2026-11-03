from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

class ImageDataset(Dataset):
    def __init__(self, csv_path: str, transform = transform):
        self.samples = []
        self.transform = transform

        self.manifest = pd.read_csv(csv_path)
        for row in self.manifest.itertuples():
            image_path = ROOT / row.Path_to_PNG
            if image_path.is_file() and row.Label != None:
                self.samples.append((row.Image_ID, image_path, row.Label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sno, image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB') # conversion to np is unsuitable for torchvision

        # print("image type before transform: ", type(image)) # type -> PIL Image object.
        if self.transform:
            image = self.transform(image)

        # print("image after transform: ", image.shape)
        # print("image type after transform: ", type(image))  # type -> torch tensor
        return sno, image, label


if __name__ == '__main__':
    print("testing dataloader")

    csv_path = ROOT / 'results' / 'artifacts' / 'train_manifest.csv'
    train_dataset = ImageDataset(csv_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)

    print("number of images: ", len(train_dataset))
    print("number of batches: ", len(train_loader))

    print("sample 1 of train_dataset: ", train_dataset[0])
