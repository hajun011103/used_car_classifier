# train.py
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau # SequentialLR, LinearLR, CosineAnnealingLR,
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import log_loss
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import datetime

import config
from dataset import full_dataset, mixup_cutmix_data
from loss import FocalLoss, compute_class_alpha
from model import build_model
from utils import save_checkpoint, load_checkpoint, setup_logging

# SEED
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # 재현성이 중요
    torch.backends.cudnn.benchmark = False     # 속도가 중요

# train loop
def train_loop(train_loader, val_loader, logger):
    # TensorBoard init
    log_dir = os.path.join("runs", config.RUN_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # have changed the runs name to runs2
    writer = SummaryWriter(log_dir=log_dir)

    # build model
    model = build_model().to(config.DEVICE)

    # Optimizer (AdamW)
    # optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    optimizer = AdamW(model.parameters(), lr=config.LR)

    # Scheduler (Reduce LR on Plateau using val loss as metrics)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, threshold=0.01, threshold_mode='rel', min_lr=1e-10)

    # warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)                               # because i dont have time i have chnaged the warmup epochs to 3 and total epochs to 15
    # scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    # cosine = CosineAnnealingLR(optimizer, T_max=config.EPOCHS-5, eta_min=config.MIN_LR)
    # scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    # Loss Function (Cross Entropy Loss function or Focal Loss)
    # criterion = nn.CrossEntropyLoss()
    alpha = compute_class_alpha()
    criterion = FocalLoss(gamma=2.0, alpha=None)
    scaler = torch.amp.GradScaler(device='cuda')

    # Resume checkpoint if exists
    # resume_epoch = 0  # indicate where i want to start from

    # # 아래 코드 수정
    # start_epoch = 0
    # if resume_epoch is not None:
    #     resume_path = os.path.join(config.SAVE_DIR, "checkpoint.pt")
    #     if os.path.exists(resume_path):
    #         logger.info(f"Loading checkpoint from {resume_path}")
    #         checkpoint = load_checkpoint(model, optimizer, filename=resume_path)
    #         start_epoch = checkpoint.get("epoch", resume_epoch) + 1

    start_epoch = 0
    best_logloss = 100
    patience_counter = 0

    for epoch in range(start_epoch, config.EPOCHS):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        use_mix = epoch < (config.EPOCHS - config.NO_MIX)

        for imgs, labels in tqdm(train_loader, f"Train Epoch{epoch+1}"):
            imgs = imgs.to(config.DEVICE, memory_format=torch.channels_last)
            labels = labels.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)

            #  Disable when use mix up and cut mix
            # with torch.amp.autocast(device_type='cuda'):
            #     logits = model(imgs)
            #     loss = criterion(logits, labels)
                
            # Mix Up & Cut Mix
            if use_mix:
                imgs, targets_a, targets_b, lam = mixup_cutmix_data(imgs, labels, alpha=1.0, cutmix_prob=0.5)
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(imgs)
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = nn.functional.softmax(logits, dim=1)

        train_loss = train_loss / len(train_loader)

        # ----- Validate -----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_probs, all_labels = [], []

        # Check Image Qulity
        class_confidences = defaultdict(list)

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(config.DEVICE, memory_format=torch.channels_last)
                labels = labels.to(config.DEVICE)
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                val_loss  += loss.item()

                # Accuracy
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # Log Loss
                probs = nn.functional.softmax(logits, dim=1)
                probs = probs / probs.sum(dim=1, keepdim=True)
                all_probs.extend(probs.cpu().numpy().astype(np.float64))
                all_labels.extend(labels.cpu().numpy())

                # True Class Probability tracking
                for i in range(labels.size(0)):
                    true_label = labels[i].item()
                    true_prob = probs[i, true_label].item()
                    class_confidences[true_label].append(true_prob)

        avg_val_loss = val_loss / len(val_loader)
        val_acc  = 100* val_correct / val_total
        val_logloss   = log_loss(all_labels, np.array(all_probs, dtype=np.float64), labels=list(range(config.NUM_CLASSES)))

        # Compute and print average confidence per class
        avg_confidence_per_class = {
            class_idx: np.mean(probs)
            for class_idx, probs in class_confidences.items()
        }

        sorted_confidence = sorted(avg_confidence_per_class.items(), key=lambda x: x[1])

        print("\n🔺 [Top 5 Classes with Highest Avg True-Label Probability]")
        for cls_idx, avg_prob in sorted_confidence[-5:][::-1]:
            print(f"{config.CLASS_NAMES[cls_idx]:40s}: {avg_prob:.4f}")

        print("\n🔻 [Top 5 Classes with Lowest Avg True-Label Probability]")
        for cls_idx, avg_prob in sorted_confidence[:5]:
            print(f"{config.CLASS_NAMES[cls_idx]:40s}: {avg_prob:.4f}")

        # ----- Scheduler Step -----
        scheduler.step(val_logloss)
        current_lr = optimizer.param_groups[0]['lr']

        # ----- Logging -----
        logger.info(
            f"Epoch{epoch+1}/{config.EPOCHS}] "
            f"LR: {current_lr} "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.3f}%, Log Loss: {val_logloss:.4f}"
        )

        # TensorBoard polt
        writer.add_scalar('Learning Rate', current_lr, epoch+1)
        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch+1)
        writer.add_scalar('Val/Accuracy', val_acc, epoch+1)
        writer.add_scalar('Val/Log_Loss', val_logloss, epoch+1)

        # ----- Checkpoint & Early Stopping -----
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            patience_counter = 0 # reset
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    save_checkpoint({
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict()
    }, is_best=True)
    
    writer.close()


def main():
    # Seed & Logger
    set_seed(config.SEED)
    logger = setup_logging(log_dir="logs", log_name="train.log")

    # Load Dataset (with Stratified Split) & Start traiing
    train_loader, val_loader = full_dataset()
    train_loop(train_loader, val_loader, logger)


if __name__ == "__main__":
    main()
