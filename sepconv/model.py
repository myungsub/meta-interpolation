import os
import sys
import math
import torch

try:
    #from sepconv.sepconv import sepconv
    from .sepconv_op import sepconv  # the custom separable convolution layer
    print('import sepconv')
except:
    #sys.path.insert(0, './sepconv_op')
    sys.path.insert(0, './sepconv/sepconv_op')
    #import sepconv  # you should consider upgrading python
    from sepconv_op import sepconv
    print('import sepconv?')
# end

from model_utils import MetaConv2dLayer, MetaSequential, extract_top_level_dict

class Network(torch.nn.Module):
    def __init__(self, resume=False, strModel='lf'):
        super(Network, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()

        self.padding = [25, 25, 25, 25]
        self.modulePad = torch.nn.ReplicationPad2d(self.padding)

        if resume:
            print('Loading model: network-%s.pytorch' % strModel)
            self.load_state_dict(torch.load('./network-' + strModel + '.pytorch'))
    # end

    def forward(self, tensorFirst, tensorSecond):
        # Do the paddings here
        width, height = tensorFirst.size(3), tensorFirst.size(2)
        intPaddingWidth = self.padding[0] + width + self.padding[1]
        intPaddingHeight = self.padding[2] + height + self.padding[3]
        if intPaddingWidth != ((intPaddingWidth >> 7) << 7):
            intPaddingWidth = (((intPaddingWidth >> 7) + 1) << 7)
        if intPaddingHeight != ((intPaddingHeight >> 7) << 7):
            intPaddingHeight = (((intPaddingHeight >> 7) + 1) << 7)
        modulePaddingInput = torch.nn.ReplicationPad2d(
            padding=[self.padding[0], intPaddingWidth - self.padding[0] - width, \
                     self.padding[2], intPaddingHeight - self.padding[2] - height]).cuda()
        modulePaddingOutput = torch.nn.ReplicationPad2d(
            padding=[0 - self.padding[0], self.padding[0] + width - intPaddingWidth, \
                     0 - self.padding[2], self.padding[2] + height - intPaddingHeight]).cuda()

        tensorPreprocessedFirst = modulePaddingInput(tensorFirst)
        tensorPreprocessedSecond = modulePaddingInput(tensorSecond)
        tensorJoin = torch.cat([ tensorPreprocessedFirst, tensorPreprocessedSecond ], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        #tensorConv1 = self.moduleConv1(tensorPreprocessed)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        tensorDot1 = sepconv.FunctionSepconv()(self.modulePad(tensorPreprocessedFirst), self.moduleVertical1(tensorCombine), self.moduleHorizontal1(tensorCombine))
        tensorDot2 = sepconv.FunctionSepconv()(self.modulePad(tensorPreprocessedSecond), self.moduleVertical2(tensorCombine), self.moduleHorizontal2(tensorCombine))

        return modulePaddingOutput(tensorDot1 + tensorDot2)
    # end
# end


class MetaNetwork(torch.nn.Module):
    def __init__(self, resume=False, strModel='lf'):
        super(MetaNetwork, self).__init__()

        def Basic(intInput, intOutput):
            return MetaSequential(
                MetaConv2dLayer(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                MetaConv2dLayer(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                MetaConv2dLayer(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return MetaSequential(
                MetaConv2dLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                MetaConv2dLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                MetaConv2dLayer(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                MetaConv2dLayer(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = MetaSequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MetaConv2dLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = MetaSequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MetaConv2dLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = MetaSequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MetaConv2dLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = MetaSequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MetaConv2dLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()

        self.padding = [25, 25, 25, 25]
        self.modulePad = torch.nn.ReplicationPad2d(self.padding)

        if resume:
            print('Loading model: pretrained_models/sepconv_base_%s.pth' % strModel)
            self.load_state_dict(torch.load('pretrained_models/sepconv_base_' + strModel + '.pth'))
    # end

    def forward(self, tensorFirst, tensorSecond, params=None, **kwargs):
        # Do the paddings here
        width, height = tensorFirst.size(3), tensorFirst.size(2)
        intPaddingWidth = self.padding[0] + width + self.padding[1]
        intPaddingHeight = self.padding[2] + height + self.padding[3]
        if intPaddingWidth != ((intPaddingWidth >> 7) << 7):
            intPaddingWidth = (((intPaddingWidth >> 7) + 1) << 7)
        if intPaddingHeight != ((intPaddingHeight >> 7) << 7):
            intPaddingHeight = (((intPaddingHeight >> 7) + 1) << 7)
        modulePaddingInput = torch.nn.ReplicationPad2d(
            padding=[self.padding[0], intPaddingWidth - self.padding[0] - width, \
                     self.padding[2], intPaddingHeight - self.padding[2] - height]).cuda()
        modulePaddingOutput = torch.nn.ReplicationPad2d(
            padding=[0 - self.padding[0], self.padding[0] + width - intPaddingWidth, \
                     0 - self.padding[2], self.padding[2] + height - intPaddingHeight]).cuda()

        tensorPreprocessedFirst = modulePaddingInput(tensorFirst)
        tensorPreprocessedSecond = modulePaddingInput(tensorSecond)
        tensorJoin = torch.cat([ tensorPreprocessedFirst, tensorPreprocessedSecond ], 1)

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

            tensorConv1 = self.moduleConv1(tensorJoin, param_dict['moduleConv1'])
            tensorPool1 = self.modulePool1(tensorConv1)

            tensorConv2 = self.moduleConv2(tensorPool1, param_dict['moduleConv2'])
            tensorPool2 = self.modulePool2(tensorConv2)

            tensorConv3 = self.moduleConv3(tensorPool2, param_dict['moduleConv3'])
            tensorPool3 = self.modulePool3(tensorConv3)

            tensorConv4 = self.moduleConv4(tensorPool3, param_dict['moduleConv4'])
            tensorPool4 = self.modulePool4(tensorConv4)

            tensorConv5 = self.moduleConv5(tensorPool4, param_dict['moduleConv5'])
            tensorPool5 = self.modulePool5(tensorConv5)

            tensorDeconv5 = self.moduleDeconv5(tensorPool5, param_dict['moduleDeconv5'])
            tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

            tensorCombine = tensorUpsample5 + tensorConv5

            tensorDeconv4 = self.moduleDeconv4(tensorCombine, param_dict['moduleDeconv4'])
            tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

            tensorCombine = tensorUpsample4 + tensorConv4

            tensorDeconv3 = self.moduleDeconv3(tensorCombine, param_dict['moduleDeconv3'])
            tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

            tensorCombine = tensorUpsample3 + tensorConv3

            tensorDeconv2 = self.moduleDeconv2(tensorCombine, param_dict['moduleDeconv2'])
            tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

            tensorCombine = tensorUpsample2 + tensorConv2
        else:
            tensorConv1 = self.moduleConv1(tensorJoin)
            tensorPool1 = self.modulePool1(tensorConv1)

            tensorConv2 = self.moduleConv2(tensorPool1)
            tensorPool2 = self.modulePool2(tensorConv2)

            tensorConv3 = self.moduleConv3(tensorPool2)
            tensorPool3 = self.modulePool3(tensorConv3)

            tensorConv4 = self.moduleConv4(tensorPool3)
            tensorPool4 = self.modulePool4(tensorConv4)

            tensorConv5 = self.moduleConv5(tensorPool4)
            tensorPool5 = self.modulePool5(tensorConv5)

            tensorDeconv5 = self.moduleDeconv5(tensorPool5)
            tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

            tensorCombine = tensorUpsample5 + tensorConv5

            tensorDeconv4 = self.moduleDeconv4(tensorCombine)
            tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

            tensorCombine = tensorUpsample4 + tensorConv4

            tensorDeconv3 = self.moduleDeconv3(tensorCombine)
            tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

            tensorCombine = tensorUpsample3 + tensorConv3

            tensorDeconv2 = self.moduleDeconv2(tensorCombine)
            tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

            tensorCombine = tensorUpsample2 + tensorConv2

        tensorDot1 = sepconv.FunctionSepconv.apply(self.modulePad(tensorPreprocessedFirst), self.moduleVertical1(tensorCombine), self.moduleHorizontal1(tensorCombine))
        tensorDot2 = sepconv.FunctionSepconv.apply(self.modulePad(tensorPreprocessedSecond), self.moduleVertical2(tensorCombine), self.moduleHorizontal2(tensorCombine))

        return modulePaddingOutput(tensorDot1 + tensorDot2)
    # end

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
# end
