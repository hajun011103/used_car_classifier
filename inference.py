# inference.py
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

import config
from dataset import get_transforms, PadSquare
from model import build_model

# Load model and checkpoint
model = build_model()
checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(config.DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    # PadSquare(value=(0, 0, 0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# Test data loading
test_df = pd.read_csv(config.TEST_CSV)
image_ids = test_df['ID'].tolist()

# Insert the Empty Classes
EMPTY_CLASS_NAMES = [
    'K5_하이브리드_3세대_2020_2023',
    'RAV4_2016_2018',
    'RAV4_5세대_2019_2024',
    '디_올뉴니로_2022_2025',
    '박스터_718_2017_2024'
]

submission_columns = config.CLASS_NAMES + EMPTY_CLASS_NAMES

# Predicting
preds = []
for image_id in tqdm(image_ids, desc='Predicting'):
    img_path = os.path.join(config.TEST_DIR, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        preds.append([0] * config.NUM_CLASSES)
        continue

    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        # Add Zeros to the Empty Classes
        zero_padding = np.zeros(len(EMPTY_CLASS_NAMES), dtype=np.float64)
        padded_probs = np.concatenate([probs, zero_padding])
        preds.append(padded_probs)


# Making a submission DataFrame
submission = pd.DataFrame(preds, columns=submission_columns)
submission.insert(0, 'ID', image_ids)
submission.to_csv(config.SUBMISSION_CSV, index=False)
print(f"✅ 제출 파일 저장 완료: {config.SUBMISSION_CSV}")

# Test Dataset Noise
'''237, 773, 797. 1107, 1573, 1770, 1825, 2641, 2682, 2718, 2756, 3113, 4697, 6365'''