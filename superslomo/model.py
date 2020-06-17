import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_utils import *


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """


        super(down, self).__init__()
        # Initialize convolutional layers.
        # self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        # self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv1 = MetaConv2dLayer(in_channels=inChannels,  out_channels=outChannels, kernel_size=filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = MetaConv2dLayer(in_channels=outChannels,  out_channels=outChannels, kernel_size=filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x, params=None):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # (Convolution + Leaky ReLU) x 2
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            x = F.leaky_relu(self.conv1(x, params=param_dict['conv1']), negative_slope = 0.1)
            x = F.leaky_relu(self.conv2(x, params=param_dict['conv2']), negative_slope = 0.1)
        else:
            x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
            x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        
        super(up, self).__init__()
        # Initialize convolutional layers.
        # self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        self.conv1 = MetaConv2dLayer(in_channels=inChannels,  out_channels=outChannels, kernel_size=3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        # self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = MetaConv2dLayer(in_channels=2 * outChannels,  out_channels=outChannels, kernel_size=3, stride=1, padding=1)
           
    def forward(self, x, skpCn, params=None):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            # Convolution + Leaky ReLU
            x = F.leaky_relu(self.conv1(x, params=param_dict['conv1']), negative_slope = 0.1)
            # Convolution + Leaky ReLU on (`x`, `skpCn`)
            x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1), params=param_dict['conv2']), negative_slope = 0.1)
        else:
            # Convolution + Leaky ReLU
            x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
            # Convolution + Leaky ReLU on (`x`, `skpCn`)
            x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)
        return x



class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        
        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)
        
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """


        x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        x  = self.up1(x, s5)
        x  = self.up2(x, s4)
        x  = self.up3(x, s3)
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = F.leaky_relu(self.conv3(x), negative_slope = 0.1)
        return x


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1. 
t = np.linspace(0.125, 0.875, 7)

def getFlowCoeff (indices, device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C00, C01, C10, C11.
    """


    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C11 = C00 = - (1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))
    return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device)

def getWarpCoeff (indices, device):
    """
    Gets coefficients used for calculating final intermediate 
    frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C0 and C1.
    """


    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C0 = 1 - t[ind]
    C1 = t[ind]
    return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device)




class SuperSloMoModel(nn.Module):
    def __init__(self, device):
        super(SuperSloMoModel, self).__init__()
        self.device = device

        self.flowComp = UNet(6, 4)
        self.arbTimeFlowIntrp = UNet(20, 5)

        self.backwarp = None


    def forward(self, I0, I1, ind):
        w, h = I0.size(3), I0.size(2)
        s = 6   # bits to shift
        padW, padH = 0, 0
        if w != ((w >> s) << s):
            padW = (((w >> s) + 1) << s) - w
        if h != ((h >> s) << s):
            padH = (((h >> s) + 1) << s) - h
        paddingInput = nn.ReflectionPad2d(padding=[padW // 2, padW - padW // 2, padH // 2, padH - padH // 2])
        paddingOutput = nn.ReflectionPad2d(padding=[0 - padW // 2, padW // 2 - padW, 0 - padH // 2, padH // 2 - padH])

        I0 = paddingInput(I0)
        I1 = paddingInput(I1)


        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]

        fCoeff = getFlowCoeff(ind, self.device)

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        if self.backwarp is None or self.backwarp.W != I0.size(3) or self.backwarp.H != I0.size(2):
            self.backwarp = backWarp(I0.size(3), I0.size(2), self.device) # make grid
        g_I0_F_t_0 = self.backwarp(I0, F_t_0)
        g_I1_F_t_1 = self.backwarp(I1, F_t_1)

        intrpOut = self.arbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1 = 1 - V_t_0

        g_I0_F_t_0_f = self.backwarp(I0, F_t_0_f)
        g_I1_F_t_1_f = self.backwarp(I1, F_t_1_f)

        wCoeff = getWarpCoeff(ind, self.device)

        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

        warped_I0, warped_I1 = self.backwarp(I0, F_1_0), self.backwarp(I1, F_0_1)

        Ft_p = paddingOutput(Ft_p)
        F_0_1, F_1_0 = paddingOutput(F_0_1), paddingOutput(F_1_0)
        g_I0_F_t_0, g_I1_F_t_1 = paddingOutput(g_I0_F_t_0), paddingOutput(g_I1_F_t_1)
        warped_I0, warped_I1 = paddingOutput(warped_I0), paddingOutput(warped_I1)
        
        #return Ft_p,                                                    # output image
        #       (F_0_1, F_1_0),                                          # bidirectional flow maps
        #       (g_I0_F_t_0, g_I1_F_t_1),                                # warped intermediate images
        #       (self.backwarp(I0, F_1_0), self.backwarp(I1, F_0_1))     # warped input image (0-1, 1-0)
        return Ft_p, \
               (F_0_1, F_1_0), \
               (g_I0_F_t_0, g_I1_F_t_1), \
               (warped_I0, warped_I1)
        #       (self.backwarp(I0, F_1_0), self.backwarp(I1, F_0_1))



class MetaUNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        
        super(MetaUNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = MetaConv2dLayer(in_channels=inChannels, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv2 = MetaConv2dLayer(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1   = up(512, 512)
        self.up2   = up(512, 256)
        self.up3   = up(256, 128)
        self.up4   = up(128, 64)
        self.up5   = up(64, 32)
        self.conv3 = MetaConv2dLayer(in_channels=32, out_channels=outChannels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, params=None):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            x  = F.leaky_relu(self.conv1(x, params=param_dict['conv1']), negative_slope = 0.1)
            s1 = F.leaky_relu(self.conv2(x, params=param_dict['conv2']), negative_slope = 0.1)
            s2 = self.down1(s1, params=param_dict['down1'])
            s3 = self.down2(s2, params=param_dict['down2'])
            s4 = self.down3(s3, params=param_dict['down3'])
            s5 = self.down4(s4, params=param_dict['down4'])
            x  = self.down5(s5, params=param_dict['down5'])
            x  = self.up1(x, s5, params=param_dict['up1'])
            x  = self.up2(x, s4, params=param_dict['up2'])
            x  = self.up3(x, s3, params=param_dict['up3'])
            x  = self.up4(x, s2, params=param_dict['up4'])
            x  = self.up5(x, s1, params=param_dict['up5'])
            x  = F.leaky_relu(self.conv3(x, params=param_dict['conv3']), negative_slope = 0.1)
        else:
            x  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
            s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
            s2 = self.down1(s1)
            s3 = self.down2(s2)
            s4 = self.down3(s3)
            s5 = self.down4(s4)
            x  = self.down5(s5)
            x  = self.up1(x, s5)
            x  = self.up2(x, s4)
            x  = self.up3(x, s3)
            x  = self.up4(x, s2)
            x  = self.up5(x, s1)
            x  = F.leaky_relu(self.conv3(x), negative_slope = 0.1)
        return x


class MetaSuperSloMo(nn.Module):
    def __init__(self, device, resume=False):
        super(MetaSuperSloMo, self).__init__()
        self.device = device

        self.flowComp = MetaUNet(6, 4)
        self.arbTimeFlowIntrp = MetaUNet(20, 5)

        self.backwarp = None

        if resume:
            print('Loading model: pretrained_models/superslomo_base.pth')
            # checkpoint = torch.load('pretrained_models/meta_superslomo.pth')
            checkpoint = torch.load('pretrained_models/superslomo_base.pth')
            self.flowComp.load_state_dict(checkpoint['state_dictFC'])
            self.arbTimeFlowIntrp.load_state_dict(checkpoint['state_dictAT'])


    def forward(self, I0, I1, ind=3, params=None, **kwargs):
        ind = ind * torch.ones(I0.size(0), dtype=int)
        w, h = I0.size(3), I0.size(2)
        s = 6   # bits to shift
        padW, padH = 0, 0
        if w != ((w >> s) << s):
            padW = (((w >> s) + 1) << s) - w
        if h != ((h >> s) << s):
            padH = (((h >> s) + 1) << s) - h
        paddingInput = nn.ReflectionPad2d(padding=[padW // 2, padW - padW // 2, padH // 2, padH - padH // 2])
        paddingOutput = nn.ReflectionPad2d(padding=[0 - padW // 2, padW // 2 - padW, 0 - padH // 2, padH // 2 - padH])

        I0 = paddingInput(I0)
        I1 = paddingInput(I1)

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

            flowOut = self.flowComp(torch.cat((I0, I1), dim=1), params=param_dict['flowComp'])
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = getFlowCoeff(ind, self.device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            if self.backwarp is None or self.backwarp.W != I0.size(3) or self.backwarp.H != I0.size(2):
                self.backwarp = backWarp(I0.size(3), I0.size(2), self.device) # make grid
            g_I0_F_t_0 = self.backwarp(I0, F_t_0)
            g_I1_F_t_1 = self.backwarp(I1, F_t_1)

            intrpOut = self.arbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1),
                params=param_dict['arbTimeFlowIntrp'])
        else:
            flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = getFlowCoeff(ind, self.device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            if self.backwarp is None or self.backwarp.W != I0.size(3) or self.backwarp.H != I0.size(2):
                self.backwarp = backWarp(I0.size(3), I0.size(2), self.device) # make grid
            g_I0_F_t_0 = self.backwarp(I0, F_t_0)
            g_I1_F_t_1 = self.backwarp(I1, F_t_1)

            intrpOut = self.arbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1 = 1 - V_t_0

        g_I0_F_t_0_f = self.backwarp(I0, F_t_0_f)
        g_I1_F_t_1_f = self.backwarp(I1, F_t_1_f)

        wCoeff = getWarpCoeff(ind, self.device)

        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

        warped_I0, warped_I1 = self.backwarp(I0, F_1_0), self.backwarp(I1, F_0_1)

        Ft_p = paddingOutput(Ft_p)
        F_0_1, F_1_0 = paddingOutput(F_0_1), paddingOutput(F_1_0)
        g_I0_F_t_0, g_I1_F_t_1 = paddingOutput(g_I0_F_t_0), paddingOutput(g_I1_F_t_1)
        warped_I0, warped_I1 = paddingOutput(warped_I0), paddingOutput(warped_I1)
        
        #return Ft_p,                                                    # output image
        #       (F_0_1, F_1_0),                                          # bidirectional flow maps
        #       (g_I0_F_t_0, g_I1_F_t_1),                                # warped intermediate images
        #       (self.backwarp(I0, F_1_0), self.backwarp(I1, F_0_1))     # warped input image (0-1, 1-0)
        return Ft_p, {
               'bidirectional_flow': (F_0_1, F_1_0),
               'warped_intermediate_frames': (g_I0_F_t_0, g_I1_F_t_1),
               'warped_input_frames': (warped_I0, warped_I1)}
        #       (self.backwarp(I0, F_1_0), self.backwarp(I1, F_0_1))
        # return Ft_p


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
