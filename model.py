# model.py
import torch
import torch.nn as nn
import timm

import config

# model = ConvNextV2-tiny, EfficientNetV2
def build_model(model=config.MODEL):

    if model == 'convnext':
        model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=config.NUM_CLASSES)
        model = model.to(config.DEVICE, memory_format=torch.channels_last)
        return model
    
    elif model == 'efficientnet':
        model = timm.create_model('efficientnet_b3a', pretrained=True, num_classes=config.NUM_CLASSES, drop_rate=0.3, drop_path_rate=0.2)
        model = model.to(config.DEVICE, memory_format=torch.channels_last)
        return model
    
    elif model == 'resnet':
        model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=config.NUM_CLASSES, drop_rate=0.3, drop_path_rate=0.2)
        model = model.to(config.DEVICE, memory_format=torch.channels_last)
        return model
    
    elif model == 'deit':
        model = timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True, num_classes=config.NUM_CLASSES, drop_rate=0.3, drop_path_rate=0.2)
        model = model.to(config.DEVICE, memory_format=torch.channels_last)
        return model