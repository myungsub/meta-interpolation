import torch
from torch import nn
import torch.nn.functional as F

# Adapted from "Tunable U-Net implementation in PyTorch"
# https://github.com/jvanvugt/pytorch-unet

from model_utils import *

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=5,
        padding=True,
    ):
        super(UNet, self).__init__()

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding)
            )
            prev_channels = 2 ** (wf + i)
        self.midconv = nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), padding)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=3,padding=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.midconv(x), negative_slope = 0.1)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetUpBlock, self).__init__()

        self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            )
        self.conv_block = UNetConvBlock(in_size, out_size, padding)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]
    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat((up, crop1), 1)
        out = self.conv_block(out)
        return out




class MetaUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=5,
        padding=True,
    ):
        super(MetaUNet, self).__init__()

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                MetaUNetConvBlock(prev_channels, 2 ** (wf + i), padding)
            )
            prev_channels = 2 ** (wf + i)
        self.midconv = MetaConv2dLayer(prev_channels, prev_channels, kernel_size=3, stride=1, padding=1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                MetaUNetUpBlock(prev_channels, 2 ** (wf + i), padding)
            )
            prev_channels = 2 ** (wf + i)

        self.last = MetaConv2dLayer(prev_channels, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, params=None):
        blocks = []

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            down_param_dict = extract_top_level_dict(current_dict=param_dict['down_path'])
            up_param_dict = extract_top_level_dict(current_dict=param_dict['up_path'])
            for i, down in enumerate(self.down_path):
                x = down(x, params=down_param_dict[str(i)])
                if i != len(self.down_path) - 1:
                    blocks.append(x)
                    x = F.avg_pool2d(x, 2)
            x = F.leaky_relu(self.midconv(x, params=param_dict['midconv']), negative_slope = 0.1)
            for i, up in enumerate(self.up_path):
                x = up(x, blocks[-i - 1], params=up_param_dict[str(i)])
            out = self.last(x, params=param_dict['last'])
        else:
            for i, down in enumerate(self.down_path):
                x = down(x)
                if i != len(self.down_path) - 1:
                    blocks.append(x)
                    x = F.avg_pool2d(x, 2)
            x = F.leaky_relu(self.midconv(x), negative_slope = 0.1)
            for i, up in enumerate(self.up_path):
                x = up(x, blocks[-i - 1])
            out = self.last(x)

        return out

class MetaUNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(MetaUNetConvBlock, self).__init__()
        block = []

        block.append(MetaConv2dLayer(in_size, out_size, kernel_size=3, stride=1, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))

        block.append(MetaConv2dLayer(out_size, out_size, kernel_size=3, stride=1, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))
        self.block = MetaSequential(*block)

    def forward(self, x, params=None):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            out = self.block(x, params=param_dict['block'])
        else:
            out = self.block(x)
        return out


class MetaUNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(MetaUNetUpBlock, self).__init__()

        self.up = MetaSequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                MetaConv2dLayer(in_size, out_size, kernel_size=3, stride=1, padding=1),
            )
        self.conv_block = MetaUNetConvBlock(in_size, out_size, padding)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge, params=None):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            up = self.up(x, params=param_dict['up'])
            crop1 = self.center_crop(bridge, up.shape[2:])
            out = torch.cat((up, crop1), 1)
            out = self.conv_block(out, params=param_dict['conv_block'])
        else:
            up = self.up(x)
            crop1 = self.center_crop(bridge, up.shape[2:])
            out = torch.cat((up, crop1), 1)
            out = self.conv_block(out)
        return out
