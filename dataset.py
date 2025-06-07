# dataset.py
import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from sklearn.model_selection import train_test_split

import config

# Making images Square
class PadSquare(ImageOnlyTransform):
    def __init__(self, border_mode=0, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, image, **params):
        h, w, c = image.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value)
        return image

    def get_transform_init_args_names(self):
        return ("border_mode", "value")

# Cut Half
class CutHalf(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        h, w, _ = image.shape
        direction = random.choice(["left", "right", "top", "bottom"])

        if direction == "left":
            return image[:, :w // 2, :]
        elif direction == "right":
            return image[:, w // 2:, :]
        elif direction == "top":
            return image[:h // 2, :, :]
        elif direction == "bottom":
            return image[h // 2:, :, :]
        else:
            return image  # fallback (shouldn't happen)

    def get_transform_init_args_names(self):
        return ()

# Transforms
def get_transforms(train:True, model=config.MODEL):

    if train:
        if model == 'convnext':
            transform = A.Compose([
                PadSquare(value=(0, 0, 0)),
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.Affine(translate_percent=(0.05, 0.05), p=0.3),
                A.Affine(shear={'x': (-3, 3), 'y': (-1, 1)}, p=0.1),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.3),
                # A.GaussNoise(std_range=(0.1, 0.25), p=0.3),
                # A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(5, 10), hole_width_range=(5, 10), fill='random_uniform', p=0.2),
                CutHalf(p=0.0001),
                # A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
                # A.VerticalFlip(p=0.2),
                # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                # A.ImageCompression(quality_range=(50, 100), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

        if model == 'efficientnet':
            transform = A.Compose([
                PadSquare(value=(0, 0, 0)),
                A.Resize(224, 224),
                A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                # A.Affine(translate_percent=(0.05, 0.05), p=0.5),
                # A.Affine(shear={'x': (-3, 3), 'y': (-1, 1)}, p=0.1),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.3),
                # A.GaussNoise(std_range=(0.1, 0.25), p=0.3),
                # A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(5, 10), hole_width_range=(5, 10), fill='random_uniform', p=0.2),
                # CutHalf(p=0.001),
                # A.VerticalFlip(p=0.2),
                # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                # A.ImageCompression(quality_range=(50, 100), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    else: # Valid & Inference
        transform = A.Compose([
            PadSquare(value=(0, 0, 0)),
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

    return transform

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            # 테스트셋: 라벨 없이 이미지 경로만 저장
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path, ))
        else:
            # 학습셋: 클래스별 폴더 구조에서 라벨 추출
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image=np.array(image))['image']
                # image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image=np.array(image))['image']
                # image = self.transform(image)
            return image, label

# Load Full Dataset
def full_dataset():
    full_dataset = CustomImageDataset(config.TRAIN_DIR, transform=get_transforms(True), is_test=False)
    print(f"Number of images: {len(full_dataset)} | Number of Classes: {config.NUM_CLASSES}")

    targets = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes

    # Stratified Split (with empty folders)
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=config.SEED
    )

    # WeightedRandomSampler
    train_labels = [targets[i] for i in train_idx]
    label_counts = np.bincount(train_labels)
    label_weights = 1.0 / label_counts
    sample_weights = [label_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=config.NUM_TIMES*len(sample_weights), replacement=True)

    # Subset + transform
    train_dataset = Subset(CustomImageDataset(config.TRAIN_DIR, transform=get_transforms(True, model=config.MODEL)), train_idx)
    val_dataset = Subset(CustomImageDataset(config.TRAIN_DIR, transform=get_transforms(False, model=config.MODEL)), val_idx)
    print(f'Number of Train Images: {len(train_dataset)}, Number of Valid Images: {len(val_dataset)}')

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,              
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6
    )
    
    return train_loader, val_loader

# Mix up & Cut mix
def mixup_cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    ''' Return mixed inputs, pairs of targets, and lambda '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    rand_index = torch.randperm(x.size(0)).to(x.device)
    y_a, y_b = y, y[rand_index]

    if np.random.rand() < cutmix_prob:
        # CutMix
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    else:
        # Mixup
        x = lam * x + (1 - lam) * x[rand_index, :]

    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    ''' Generate random bbox '''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2