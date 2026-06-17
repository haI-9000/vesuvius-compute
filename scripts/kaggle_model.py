import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segmentation_models_pytorch.base.modules import DecoderBlock
except ImportError:
    try:
        from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
    except ImportError:
        class DecoderBlock(nn.Module):
            def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
                self.relu1 = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
                self.relu2 = nn.ReLU(inplace=True)

            def forward(self, x, skip):
                if skip is not None:
                    x = torch.cat([x, skip], dim=1)
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                return x

try:
    from timm.models.resnet import resnet34d, resnet10t
except ImportError:
    from torchvision.models import resnet34, resnet18
    def resnet34d(pretrained=False, in_chans=3):
        model = resnet34(pretrained=pretrained)
        if in_chans != 3:
            model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    def resnet10t(pretrained=False, in_chans=3):
        model = resnet18(pretrained=pretrained)
        if in_chans != 3:
            model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

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
    def __init__(self):
        super().__init__()
        self.encoder1 = resnet34d(pretrained=False, in_chans=16)
        self.decoder1 = SmpUnetDecoder(
            in_channel=512,
            skip_channel=[64, 64, 128, 256],
            out_channel=[256, 128, 64, 64]
        )
        self.logit1 = nn.Conv2d(64, 1, kernel_size=1)

        self.encoder2 = resnet10t(pretrained=False, in_chans=64)
        self.decoder2 = SmpUnetDecoder(
            in_channel=512,
            skip_channel=[64, 128, 256],
            out_channel=[128, 64, 32]
        )
        self.logit2 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, batch):
        v = batch['volume']
        B, C, H, W = v.shape

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

        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last = self.decoder1(feature, skip)

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

        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last = self.decoder2(feature, skip)
        logit2 = self.logit2(last)
        logit2 = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False)

        return {'ink': torch.sigmoid(logit2)}
