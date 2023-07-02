# import multiprocessing
import os
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A


class CustomImageDataset(Dataset):
    def __init__(
        self,
        folder,
        csv_file,
        transform=None,
        train=True,
        test_size=0.2,
        random_state=42,
        albumentations=False,
    ):
        self.folder = folder
        self.annotations = pd.read_csv(csv_file, index_col=0)
        self.transform = transform
        self.albumentations = albumentations

        x_data = self.annotations.iloc[:, 0]
        y_data = self.annotations.iloc[:, 1]

        # Extract unique labels and create the mapping
        self.label_set = sorted(y_data.unique())
        self.label_to_int = {label: i for i, label in enumerate(self.label_set)}

        # Apply the mapping to the labels
        y_data = y_data.map(self.label_to_int)

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=test_size,
            random_state=random_state,
            stratify=y_data,
        )

        if train:
            self.x_data, self.y_data = x_train.values, y_train.values
        else:
            self.x_data, self.y_data = x_test.values, y_test.values
            

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder, self.x_data[index])
        ## Add a better path doesnt exist error handling
        image = cv2.imread(img_path)     
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.y_data[index]

        if self.transform:
            if self.albumentations:
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(transforms.ToPILImage()(image))

        return image, label


def calculate_mean_std(dataset, batch_size=64, albumentations=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    total_imgs = 0

    for batch, _ in tqdm(dataloader, desc="Calculating mean and std"):
        batch = batch.to(device)
        if albumentations:
            batch = batch.to(torch.float32) / 255.0
        current_batch_size = batch.shape[0]
        total_imgs += current_batch_size

        mean += batch.view(3, -1).mean(dim=1) * current_batch_size
        std += batch.view(3, -1).std(dim=1) * current_batch_size

    mean /= total_imgs
    std /= total_imgs

    return mean.cpu(), std.cpu()

def save_mean_std_to_file(mean, std, file_path):
    with open(file_path, 'w') as file:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, file)

def load_mean_std_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return torch.tensor(data['mean']), torch.tensor(data['std'])


def create_data_loaders(
    folder,
    csv_file,
    batch_size=64,
    test_size=0.2,
    random_state=42,
    train_transforms=None,
    val_transforms=None,
    albumentations=False,
):
    if train_transforms is None:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    if val_transforms is None:
        val_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    train_dataset = CustomImageDataset(
        folder,
        csv_file,
        transform=train_transforms,
        train=True,
        test_size=test_size,
        random_state=random_state,
        albumentations=albumentations,
    )
    val_dataset = CustomImageDataset(
        folder,
        csv_file,
        transform=val_transforms,
        train=False,
        test_size=test_size,
        random_state=random_state,
        albumentations=albumentations,
    )

    # Create label mapping from int to label
    int_to_label = {i: label for label, i in train_dataset.label_to_int.items()}

    # Check for existing mean and std values
    parent_directory = os.path.dirname(folder)
    config_file = os.path.join(parent_directory, 'mean_std_config.json')
    if os.path.exists(config_file):
        mean, std = load_mean_std_from_file(config_file)
    else:
        # Calculate mean and std for the train dataset
        mean, std = calculate_mean_std(train_dataset, batch_size=512, albumentations=albumentations)
        save_mean_std_to_file(mean, std, config_file)


    if albumentations:
        # Add normalization to the albumentations transforms
        train_transforms.transforms.insert(-1, A.Normalize(mean=mean, std=std))
        val_transforms.transforms.insert(-1, A.Normalize(mean=mean, std=std))
    else:
        # Add normalization to torchvision transforms
        train_transforms = transforms.Compose(
            [train_transforms, transforms.Normalize(mean=mean, std=std)]
        )
        val_transforms = transforms.Compose(
            [val_transforms, transforms.Normalize(mean=mean, std=std)]
        )

    # Apply updated transforms
    train_dataset.transform = train_transforms
    val_dataset.transform = val_transforms

    # Get the number of available cores
    # num_workers = multiprocessing.cpu_count()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, int_to_label
