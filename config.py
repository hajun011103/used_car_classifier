# config.py
import os

# Path
DATA_DIR         = '/home/wfscontrol/Downloads/used_cars'
TRAIN_DIR        = '/home/wfscontrol/Downloads/used_cars/train'
TEST_DIR         = '/home/wfscontrol/Downloads/used_cars/test'
SUBMISSION_CSV   = '/home/wfscontrol/used_car_classifier/convnext/submission.csv'
TEST_CSV         = os.path.join(DATA_DIR, "test.csv")
SAVE_DIR         = '/home/wfscontrol/used_car_classifier/convnext'

# Model
MODEL            = 'convnext'
# MODEL            = 'efficientnet'
# MODEL            = 'resnet'
# MODEL            = 'deit'
CHECKPOINT_PATH  = os.path.join(SAVE_DIR, f"{MODEL}.pt")

# Hyperparameters
if MODEL == 'convnext':
    BATCH_SIZE       = 58
    SEED             = 42
elif MODEL == 'efficientnet':
    BATCH_SIZE       = 58
    SEED             = 21
elif MODEL == 'resnet':
    BATCH_SIZE       = 96
    SEED             = 63
elif MODEL == 'deit':
    BATCH_SIZE       = 64
    SEED             = 84

EPOCHS           = 20
LR               = 1e-4
MIN_LR           = 1e-6
WEIGHT_DECAY     = 1e-5
PATIENCE         = 3
NUM_TIMES        = 1  # * of Augmented Data 
NO_MIX           = 2  # stop Cut Mix & Mix Up

# Tensorboard settings
RUN_NAME         = MODEL

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Names & Number of Classes
from torchvision.datasets import ImageFolder
dataset = ImageFolder(root=TRAIN_DIR)
CLASS_NAMES = dataset.classes
NUM_CLASSES = len(CLASS_NAMES)