
#[Super SloMo]
##High Quality Estimation of Multiple Intermediate Frames for Video Interpolation

import os
import copy
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloader
from math import log10
import datetime
import time
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import cv2
import numpy as np


# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True, help='name of the experiment run')
parser.add_argument("--dataset_root", type=str, required=True, help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--train_continue", type=bool, default=False, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=8, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=1, help='batch size for validation. Default: 10.')
parser.add_argument("--num_inner_update", type=int, default=1, help='set inner loop update iterations. Default: 1.')
parser.add_argument("--inner_lr", type=float, default=0.00001, help='set inner loop lr. Default: 0.00001.')
parser.add_argument("--outer_lr", type=float, default=0.00001, help='set outer loop lr. Default: 0.00001.')
parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=100, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=1, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 1.')
parser.add_argument("--mode", type=str, default='train')
args = parser.parse_args()

##[TensorboardX](https://github.com/lanpa/tensorboardX)
### For visualizing loss and interpolated frames


if not os.path.exists('log/%s' % args.exp_name):
    os.makedirs('log/%s' % args.exp_name)
writer = SummaryWriter('log/%s' % args.exp_name)


###Initialize flow computation and arbitrary-time flow interpolation CNNs.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#flowComp = model.UNet(6, 4)
#flowComp.to(device)
#ArbTimeFlowIntrp = model.UNet(20, 5)
#ArbTimeFlowIntrp.to(device)
Network = model.SuperSloMoModel(device)
Network.to(device)


###Initialze backward warpers for train and validation datasets

#trainFlowBackWarp      = model.backWarp(256, 256, device) #model.backWarp(352, 352, device)
#trainFlowBackWarp      = trainFlowBackWarp.to(device)
#validationFlowBackWarp = model.backWarp(448, 256, device) #model.backWarp(640, 352, device)
#validationFlowBackWarp = validationFlowBackWarp.to(device)


###Load Datasets


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

#trainset = dataloader.SuperSloMo(root=args.dataset_root + '/train', transform=transform, train=True)
trainset = dataloader.VimeoSeptuplet(root=args.dataset_root, transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)

#validationset = dataloader.SuperSloMo(root=args.dataset_root + '/validation', transform=transform, randomCropSize=(640, 352), train=False)
validationset = dataloader.VimeoSeptuplet(root=args.dataset_root, transform=transform, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)

print(trainset, validationset)


###Create transform to display image from tensor


negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


###Utils
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


###Loss and Optimizer


L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()

#params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())

#optimizer = optim.Adam(params, lr=args.init_learning_rate)
optimizer = optim.Adam(Network.parameters(), lr=args.outer_lr)
# scheduler to decrease learning rate by a factor of 10 at milestones.
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)


###Initializing VGG16 model for perceptual loss


vgg16 = torchvision.models.vgg16()
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
		param.requires_grad = False


### Validation function
# 
VIZ = True

def validate():
    len_validationloader = len(validationloader)
    # For details see training.
    psnr = 0
    tloss = 0
    flag = 1
    forward_times = []
    #with torch.no_grad():
    if True:
        #for validationIndex, (validationData, validationFrameIndex) in enumerate(tqdm(validationloader), 0):
        for validationIndex, (validationData, validationFrameIndex, imgpaths) in enumerate(tqdm(validationloader), 0):
            #if validationIndex == 100:
            #    len_validationloader = 100
            #    break

            weights_before = copy.deepcopy(Network.state_dict())
            val_optimizer = optim.Adam(Network.parameters(), lr=args.inner_lr)

            # inner loop
            t = time.time()
            k = args.num_inner_update
            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                total_loss = 0
                for ind in indices:
                    frame0, frameT, frame1 = validationData[ind[0]], validationData[ind[1]], validationData[ind[2]]
                    I0 = frame0.to(device)
                    I1 = frame1.to(device)
                    IFrame = frameT.to(device)
                    
                    output, (F_0_1, F_1_0), (I_0_t, I_1_t), (I_0_1, I_1_0) = Network(I0, I1, validationFrameIndex)
                    recnLoss = L1_lossFn(output, IFrame)
                    prcpLoss = MSE_LossFn(vgg16_conv_4_3(output), vgg16_conv_4_3(IFrame))
                    warpLoss = L1_lossFn(I_0_t, IFrame) + L1_lossFn(I_1_t, IFrame) + L1_lossFn(I_0_1, I1) + L1_lossFn(I_1_0, I0)
        
                    loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
                    loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
                    loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

                    loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
                    total_loss = total_loss + loss

                val_optimizer.zero_grad()
                total_loss.backward()
                val_optimizer.step()
            
            # outer loop
            frame0, frameT, frame1 = validationData[2], validationData[3], validationData[4]
            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)
            
            output, (F_0_1, F_1_0), (I_0_t, I_1_t), (I_0_1, I_1_0) = Network(I0, I1, validationFrameIndex)

            # Restore base model
            Network.load_state_dict(weights_before)


            if VIZ:
                for b in range(frame0.size(0)):
                    imgpath = imgpaths[0][b]
                    savepath = os.path.join('checkpoint', args.exp_name, 'vimeoSeptuplet', imgpath.split('/')[-3], imgpath.split('/')[-2])
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    img_pred = (revNormalize(output[b]).data.permute(1, 2, 0).clamp_(0, 1).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(savepath, 'im2_pred.png'), img_pred)


            forward_times.append(time.time() - t)
            
            # For tensorboard
            if (flag):
                retImg = torchvision.utils.make_grid([revNormalize(frame0[0]), revNormalize(frameT[0]), revNormalize(output.cpu()[0]), revNormalize(frame1[0])], padding=10)
                flag = 0
            
            #loss
            recnLoss = L1_lossFn(output, IFrame)
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(output), vgg16_conv_4_3(IFrame))
            warpLoss = L1_lossFn(I_0_t, IFrame) + L1_lossFn(I_1_t, IFrame) + L1_lossFn(I_0_1, I1) + L1_lossFn(I_1_0, I0)
        
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
            
            
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            tloss += loss.item()
            
            #psnr
            MSE_val = MSE_LossFn(output, IFrame)
            psnr += (10 * log10(1 / MSE_val.item()))

    print(np.mean(forward_times))
    return (psnr / len_validationloader), (tloss / len_validationloader), retImg


### Initialization


if args.train_continue:
    dict1 = torch.load(args.checkpoint)
    #ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    #flowComp.load_state_dict(dict1['state_dictFC'])
    Network.flowComp.load_state_dict(dict1['state_dictFC'])
    Network.arbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    print('loaded pretrained model: %s' % args.checkpoint)
else:
    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}


### Training


import time

start = time.time()
cLoss   = dict1['loss']
valLoss = dict1['valLoss']
valPSNR = dict1['valPSNR']
checkpoint_counter = 0

### Main training loop
for epoch in range(dict1['epoch'] + 1, args.epochs):
    print("Epoch: ", epoch)
        
    # Append and reset
    cLoss.append([])
    #valLoss.append([])
    #valPSNR.append([])
    iLoss = 0
    
    if args.mode == 'test':
        psnr, vLoss, valImg = validate()
        print("PSNR: %f,    Val. loss: %f" % (psnr, vLoss))
        
        break
    # Increment scheduler count    
    #scheduler.step()
    
    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
        if trainIndex == 1000:
            break


        # Meta training
        k = args.num_inner_update

        if True:    # MAML
            base_net = copy.deepcopy(Network)
            inner_optimizer = optim.Adam(Network.parameters(), lr=args.inner_lr)

            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                total_loss = 0
                for ind in indices:
                    frame0, frameT, frame1 = validationData[ind[0]], validationData[ind[1]], validationData[ind[2]]
                    I0 = frame0.to(device)
                    I1 = frame1.to(device)
                    IFrame = frameT.to(device)
                    
                    output, (F_0_1, F_1_0), (I_0_t, I_1_t), (I_0_1, I_1_0) = Network(I0, I1, validationFrameIndex)
                    recnLoss = L1_lossFn(output, IFrame)
                    prcpLoss = MSE_LossFn(vgg16_conv_4_3(output), vgg16_conv_4_3(IFrame))
                    warpLoss = L1_lossFn(I_0_t, IFrame) + L1_lossFn(I_1_t, IFrame) + L1_lossFn(I_0_1, I1) + L1_lossFn(I_1_0, I0)
        
                    loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
                    loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
                    loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

                    loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
                    total_loss = total_loss + loss

                val_optimizer.zero_grad()
                total_loss.backward()
                val_optimizer.step()
        
    		## Getting the input and the target from the training set
            #frame0, frameT, frame1 = trainData
            frame0, frameT, frame1 = trainData[2], trainData[3], trainData[4]
        
            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)
        
            output, (F_0_1, F_1_0), (I_0_t, I_1_t), (I_0_1, I_1_0) = Network(I0, I1, trainFrameIndex)
        
            recnLoss = L1_lossFn(output, IFrame)
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(output), vgg16_conv_4_3(IFrame))
            warpLoss = L1_lossFn(I_0_t, IFrame) + L1_lossFn(I_1_t, IFrame) + L1_lossFn(I_0_1, I1) + L1_lossFn(I_1_0, I0)
        
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
          
            # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
            # since the loss in paper is calculated for input pixels in range 0-255
            # and the input to our network is in range 0-1
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

            # Copy base parameters to 'Network' to connect the computation graph
            for param, base_param in zip(Network.parameters(), base_net.parameters()):
                param.data = base_param.data
        
            # Calculate gradient & update meta-learner
            optimizer.zero_grad()
            grads = torch.autograd.grad(loss, Network.parameters())
            for j, param in enumerate(Network.parameters()):
                param.grad = grads[j]
            #loss.backward()
            optimizer.step()
            iLoss += loss.item()
               
        # Logging progress every `args.progress_iter` iterations
        if ((trainIndex % args.progress_iter) == 0):#args.progress_iter):
            end = time.time()
            
            #Tensorboard
            itr = trainIndex + epoch * (len(trainloader))
            writer.add_scalars('Loss', {'trainLoss': iLoss/args.progress_iter}, itr)#,
            
            print(" Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  LearningRate: %f" % (iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, get_lr(optimizer)))
            
            cLoss[epoch].append(iLoss/args.progress_iter)
            iLoss = 0
            start = time.time()

    psnr, vLoss, valImg = validate()
            
    valPSNR.append(psnr)
    valLoss.append(vLoss)

    # Increment scheduler count    
    scheduler.step(vLoss)

    itr = epoch * len(trainloader)
    writer.add_scalars('Loss', {'validationLoss': vLoss}, itr)
    writer.add_scalar('PSNR', psnr, itr)
    writer.add_image('Validation', valImg, itr)
    print("Validation Loss: %0.6f,  PSNR: %.4f" % (vLoss, psnr))
    
    # Create checkpoint after every `args.checkpoint_epoch` epochs
    if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):
        dict1 = {
                'Detail':"End to end Super SloMo.",
                'epoch':epoch,
                'timestamp':datetime.datetime.now(),
                'trainBatchSz':args.train_batch_size,
                'validationBatchSz':args.validation_batch_size,
                'learningRate':get_lr(optimizer),
                'loss':cLoss,
                'valLoss':valLoss,
                'valPSNR':valPSNR,
                #'state_dictFC': flowComp.state_dict(),
                #'state_dictAT': ArbTimeFlowIntrp.state_dict(),
                'state_dictFC': Network.flowComp.state_dict(),
                'state_dictAT': Network.arbTimeFlowIntrp.state_dict(),
                }
        torch.save(dict1, args.checkpoint_dir + "/SuperSloMo" + str(checkpoint_counter) + ".ckpt")
        checkpoint_counter += 1
