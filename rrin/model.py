import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .unet import UNet, MetaUNet
from model_utils import extract_top_level_dict

def warp(img, flow):
    _, _, H, W = img.size()
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False).cuda()
    gridY = torch.tensor(gridY, requires_grad=False).cuda()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]
    x = gridX.unsqueeze(0).expand_as(u).float()+u
    y = gridY.unsqueeze(0).expand_as(v).float()+v
    normx = 2*(x/W-0.5)
    normy = 2*(y/H-0.5)
    grid = torch.stack((normx,normy), dim=3)
    warped = F.grid_sample(img, grid, align_corners=False)
    return warped

class Net(nn.Module):
    def __init__(self,level=3):
        super(Net, self).__init__()
        self.Mask = UNet(16,2,4)
        self.Flow_L = UNet(6,4,5)
        self.refine_flow = UNet(10,4,4)
        self.final = UNet(9,3,4)

    def process(self,x0,x1,t):

        x = torch.cat((x0,x1),1)
        Flow = self.Flow_L(x)
        Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
        Flow_t_0 = -(1-t)*t*Flow_0_1+t*t*Flow_1_0
        Flow_t_1 = (1-t)*(1-t)*Flow_0_1-t*(1-t)*Flow_1_0
        Flow_t = torch.cat((Flow_t_0,Flow_t_1,x),1)
        Flow_t = self.refine_flow(Flow_t)
        Flow_t_0 = Flow_t_0+Flow_t[:,:2,:,:]
        Flow_t_1 = Flow_t_1+Flow_t[:,2:4,:,:]
        xt1 = warp(x0,Flow_t_0)
        xt2 = warp(x1,Flow_t_1)
        temp = torch.cat((Flow_t_0,Flow_t_1,x,xt1,xt2),1)
        Mask = F.sigmoid(self.Mask(temp))
        w1, w2 = (1-t)*Mask[:,0:1,:,:], t*Mask[:,1:2,:,:]
        output = (w1*xt1+w2*xt2)/(w1+w2+1e-8)

        return output

    def forward(self, input0, input1, t=0.5):

        output = self.process(input0,input1,t)
        compose = torch.cat((input0, input1, output),1)
        final = self.final(compose)+output
        final = final.clamp(0,1)

        return final


class MetaRRIN(nn.Module):
    def __init__(self, level=3, resume=False):
        super(MetaRRIN, self).__init__()
        self.Mask = MetaUNet(16,2,4)
        self.Flow_L = MetaUNet(6,4,5)
        self.refine_flow = MetaUNet(10,4,4)
        self.final = MetaUNet(9,3,4)

        if resume:
            print('Loading model: pretrained_models/rrin_base.pth')
            checkpoint = torch.load('pretrained_models/rrin_base.pth')
            self.load_state_dict(checkpoint) #['state_dict'])

    def process(self, x0, x1, t, params=None):

        x = torch.cat((x0, x1), 1)

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            Flow = self.Flow_L(x, param_dict['Flow_L'])
            Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
            Flow_t_0 = -(1-t) * t * Flow_0_1 + t * t * Flow_1_0
            Flow_t_1 = (1-t) * (1-t) * Flow_0_1 - t * (1-t) * Flow_1_0
            Flow_t = torch.cat((Flow_t_0, Flow_t_1,x), 1)
            Flow_t = self.refine_flow(Flow_t, param_dict['refine_flow'])
        else:
            Flow = self.Flow_L(x)
            Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
            Flow_t_0 = -(1-t) * t * Flow_0_1 + t * t * Flow_1_0
            Flow_t_1 = (1-t) * (1-t) * Flow_0_1 - t * (1-t) * Flow_1_0
            Flow_t = torch.cat((Flow_t_0, Flow_t_1,x), 1)
            Flow_t = self.refine_flow(Flow_t)

        Flow_t_0 = Flow_t_0 + Flow_t[:,:2,:,:]
        Flow_t_1 = Flow_t_1 + Flow_t[:,2:4,:,:]
        xt1 = warp(x0, Flow_t_0)
        xt2 = warp(x1, Flow_t_1)
        temp = torch.cat((Flow_t_0, Flow_t_1, x, xt1, xt2),1)
        #Mask = F.sigmoid(self.Mask(temp))
        Mask = torch.sigmoid(self.Mask(temp))
        w1, w2 = (1-t) * Mask[:,0:1,:,:], t * Mask[:,1:2,:,:]
        output = (w1 * xt1 + w2 * xt2) / (w1 + w2 + 1e-8)

        return output

    def forward(self, input0, input1, t=0.5, params=None, **kwargs):

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            output = self.process(input0, input1, t, params=params)
            compose = torch.cat((input0, input1, output),1)
            final = self.final(compose, params=param_dict['final']) + output
            final = final.clamp(0,1)
        else:
            output = self.process(input0, input1, t)
            compose = torch.cat((input0, input1, output),1)
            final = self.final(compose)+output
            final = final.clamp(0,1)

        return final


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
                            param[name].grad = None

    def restore_backup_stats(self):
        pass
