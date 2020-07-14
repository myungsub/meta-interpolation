import math
import numpy as np

import torch
import torch.nn as nn

from model_utils import *


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, True)
        
        # FF_RCAN or FF_Resblocks
        self.interpolate = MetaInterpolation(5, 12, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2, params=None):
        """
        Encoder: Shuffle-spread --> Interpolate --> Return (shuffled) interpolated frame
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            feats = self.interpolate(feats1, feats2, params=params['interpolate'])
        else:
            feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats, params=None):
        out = self.shuffler(feats)
        return out


class MetaCAIN(nn.Module):
    def __init__(self, depth=3, resume=False):
        super(MetaCAIN, self).__init__()
        
        self.encoder = Encoder(in_channels=3, depth=depth)
        self.decoder = Decoder(depth=depth)

        if resume:
            print('Loading model: pretrained_models/cain_base.pth')
            checkpoint = torch.load('pretrained_models/cain_base.pth')
            #checkpoint = torch.load('pretrained_models/CAIN_meta_best.pth')
            # checkpoint = torch.load('checkpoint/CAIN_ft/model_best.pth')
            pretrained_state_dict = {
                key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
            self.load_state_dict(pretrained_state_dict)
            del checkpoint, pretrained_state_dict

    def forward(self, x1, x2, params=None, **kwargs):
        x1, m1 = sub_mean(x1)  # Using an inplace sub_mean results in errors
        x2, m2 = sub_mean(x2)

        if True: # not self.training:
            paddingInput, paddingOutput = InOutPaddings(x1)
            x1 = paddingInput(x1)
            x2 = paddingInput(x2)

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            feats = self.encoder(x1, x2, params=param_dict['encoder'])
            out = self.decoder(feats, None) # no parameters in decoder - just pixelshuffle
        else:
            feats = self.encoder(x1, x2)
            out = self.decoder(feats)

        if True: #not self.training:
            out = paddingOutput(out)

        mi = (m1 + m2) / 2
        out += mi

        return out #, feats


    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        pass    # no batch statistics used
