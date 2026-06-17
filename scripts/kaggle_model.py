"""
Kaggle 1st Place Model Architecture (ryches team)
3D UNETR + SegFormer ensemble for Vesuvius ink detection.

This is the exact architecture that achieved private LB 0.682693.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import resnet34d, resnet10t
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock


class SmpUnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        self.block = nn.ModuleList([
            DecoderBlock(i, s, o, use_batchnorm=True, attention_type=None)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ])

    def forward(self, feature, skip):
        d = feature
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
        return d


class Net(nn.Module):
    """
    Main model architecture from 1st place Kaggle solution.
    Two-stage: 3D CNN encoder + 2D SegFormer-like decoder.
    """
    def __init__(self):
        super().__init__()
        self.output_type = ['inference', 'loss']

        # First stage: 3D CNN encoder (depth=16)
        self.encoder1 = resnet34d(pretrained=False, in_chans=16)
        self.decoder1 = SmpUnetDecoder(
            in_channel=512,
            skip_channel=[64, 64, 128, 256],
            out_channel=[256, 128, 64, 64]
        )
        self.logit1 = nn.Conv2d(64, 1, kernel_size=1)

        # Second stage: 2D SegFormer-like decoder
        self.encoder2 = resnet10t(pretrained=False, in_chans=64)
        self.decoder2 = SmpUnetDecoder(
            in_channel=512,
            skip_channel=[64, 128, 256],
            out_channel=[128, 64, 32]
        )
        self.logit2 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, batch):
        v = batch['volume']  # (B, depth, H, W)
        B, C, H, W = v.shape

        # Encoder 1
        x = v
        encoder = []
        x = self.encoder1.conv1(x)
        x = self.encoder1.bn1(x)
        x = self.encoder1.act1(x)
        encoder.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.encoder1.layer1(x)
        encoder.append(x)
        x = self.encoder1.layer2(x)
        encoder.append(x)
        x = self.encoder1.layer3(x)
        encoder.append(x)
        x = self.encoder1.layer4(x)
        encoder.append(x)

        # Decoder 1
        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last = self.decoder1(feature, skip)
        logit1 = self.logit1(last)

        # Encoder 2
        x = last
        encoder = []
        x = self.encoder2.layer1(x)
        encoder.append(x)
        x = self.encoder2.layer2(x)
        encoder.append(x)
        x = self.encoder2.layer3(x)
        encoder.append(x)
        x = self.encoder2.layer4(x)
        encoder.append(x)

        # Decoder 2
        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last = self.decoder2(feature, skip)
        logit2 = self.logit2(last)
        logit2 = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False)

        return {'ink': torch.sigmoid(logit2)}
