#!/usr/bin/env python

import getopt
import math
import copy
import cv2
import numpy
import os
import time
import argparse
import torch
#import torch.utils.serialization
from tqdm import tqdm
import random

# from sepconv.model import Network           # model
from sepconv.model import MetaNetwork as Network           # model
#from combined240 import Combined240 # data
from data.vimeo_septuplet import VimeoSeptuplet # data
import utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

VIZ = True

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='temp', help='Name of the experiment')
    parser.add_argument('--meta_algorithm', type=str, default='maml', help='Meta-learning algorithm to use')
    parser.add_argument('--batch_size', '--bs', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--inner_lr', type=float, default=0.00001)
    parser.add_argument('--outer_lr', type=float, default=0.00001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--val_iter', type=int, default=200)
    parser.add_argument('--resume', action='store_true', help='True if resuming from a pretrained model')
    parser.add_argument('--resume_model', type=str, default='l1')
    parser.add_argument('--resume_ckpt', action='store_true')
    parser.add_argument('--resume_exp', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--logfreq', type=int, default=20)
    parser.add_argument('--num_inner_update', type=int, default=1)
    parser.add_argument('--mode', type=str, default='')
    args = parser.parse_args()

    # Train args
    args.step = 3
    args.crop_size = [128, 128]
    args.rotation = [-10, 10]
    args.crop_policy = 'random'
    args.flip = True
    args.scale_factor = [1.07, 1.5]

    return args

##########################################################
cfg = parse_args()
VAL_ITER_CUT = 1e6 if cfg.val_iter==-1 else cfg.val_iter

net = Network(cfg.resume, cfg.resume_model).cuda() #.eval()

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adamax(net.parameters(), lr=cfg.outer_lr)

if cfg.resume_ckpt:
    utils.load_checkpoint(cfg, net, optimizer)
    '''
    print('loading pretrained checkpoint.. %s' % cfg.exp_name)
    #checkpoint = torch.load(os.path.join('checkpoint', cfg.exp_name, 'checkpoint.pth'))
    checkpoint = torch.load(os.path.join('checkpoint', cfg.exp_name, 'model_best.pth'))
    cfg.start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['grad_dict'])
    del checkpoint'''

##########################################################

train_loader = torch.utils.data.DataLoader(#Combined240('data/combined', cfg),
    #VimeoSeptuplet('data/vimeo_septuplet', [1, 2, 3, 4, 5, 6, 7], is_training=True),
    VimeoSeptuplet(cfg),
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=False
)


def train(epoch):
    print("Training Epoch: %d" % epoch)
    net.train()

    losses = utils.AverageMeter()
    psnrs = utils.AverageMeter()
    ssims = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    t = time.time()
    for i, (images, _) in enumerate(train_loader):
        if i == 4000:
            break
        images = [im.cuda() for im in images]


        # Meta training
        k = cfg.num_inner_update   # inner update iteration

        if cfg.meta_algorithm == 'reptile':
            weights_before = copy.deepcopy(net.state_dict())
            inner_optimizer = torch.optim.Adamax(net.parameters(), lr=cfg.inner_lr)

            # inner loop
            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                total_loss = 0
                for ind in indices:
                    output = net(images[ind[0]].clone(), images[ind[2]].clone())
                    loss = criterion(output, images[ind[1]])
                    total_loss = total_loss + loss

                inner_optimizer.zero_grad()
                total_loss.backward()
                inner_optimizer.step()

            # Reptile - outer update
            outerstepsize = cfg.outer_lr
            weights_after = net.state_dict()
            net.load_state_dict({name:
                weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                for name in weights_before})

            # calculate loss w/ updated model
            input0, input1, target = images[2], images[4], images[3]

            with torch.no_grad():
                output = net(input0, input1)
                loss = criterion(output, target)


        elif cfg.meta_algorithm == 'maml':

            base_net = copy.deepcopy(net)
            inner_optimizer = torch.optim.Adamax(net.parameters(), lr=cfg.inner_lr)

            # inner loop
            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                total_loss = 0
                for ind in indices:
                    output = net(images[ind[0]].clone(), images[ind[2]].clone())
                    loss = criterion(output, images[ind[1]])
                    total_loss = total_loss + loss

                inner_optimizer.zero_grad()
                total_loss.backward()
                inner_optimizer.step()

            # Forward on query data
            outerstepsize = cfg.outer_lr
            input0, input1, target = images[2], images[4], images[3]
            output = net(input0, input1)
            loss = criterion(output, target)

            # Copy base parameters to 'net' to connect the computation graph
            for param, base_param in zip(net.parameters(), base_net.parameters()):
                param.data = base_param.data

            # Calculate gradient & update meta-learner
            optimizer.zero_grad()
            grads = torch.autograd.grad(loss, net.parameters())
            for j, param in enumerate(net.parameters()):
                #param = param - grads[j] * outerstepsize
                param.grad = grads[j]
            optimizer.step()

        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - t)
        t = time.time()

        # Logging
        if i % cfg.logfreq == 0:
            utils.eval_metrics(output, target, psnrs, ssims, None)
            print(('Epoch: [%d][%d/%d],\tTime %.3f (%.3f)\tLoss %.4f (%.4f)\tPSNR %.2f\t'
                   % (epoch, i, len(train_loader), batch_time.val, batch_time.avg,
                      losses.val, losses.avg, psnrs.val)))
            losses.reset()
            psnrs.reset()
            ssims.reset()
            batch_time.reset()


def validate(model, epoch):
    val_dataset = VimeoSeptuplet(cfg) #('data/vimeo_septuplet', [1, 2, 3, 4, 5, 6, 7], is_training=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    #model.eval()
    if True: #with torch.no_grad():
        losses = utils.AverageMeter()
        psnrs = utils.AverageMeter()
        ssims = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        t = time.time()

        for i, (images, metadata) in enumerate(tqdm(val_loader)):
            if i == VAL_ITER_CUT:
                break
            t = time.time()
            images = [im.cuda() for im in images]
            target = images[3]
        
            weights_before = copy.deepcopy(net.state_dict())
            val_optimizer = torch.optim.Adamax(net.parameters(), lr=cfg.inner_lr)
        
            # inner loop
            k = cfg.num_inner_update
            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                total_loss = 0
                for ind in indices:
                    output = net(images[ind[0]].clone(), images[ind[2]].clone())
                    loss = criterion(output, images[ind[1]])
                    total_loss = total_loss + loss

                val_optimizer.zero_grad()
                total_loss.backward()
                val_optimizer.step()



            with torch.no_grad():
                input0, input1, target = images[2], images[4], images[3]
                output = model(input0, input1)
                
                batch_time.update(time.time() - t)
                loss = criterion(output, target)
                losses.update(loss.item(), input0.size(0))

                utils.eval_metrics(output, target, psnrs, ssims, None)
                print(psnrs.val, ssims.val)

                if VIZ:
                    for b in range(images[0].size(0)):
                        imgpath = metadata['imgpaths'][0][b]
                        #print(imgpath)
                        savepath = os.path.join('checkpoint', cfg.exp_name, 'vimeoSeptuplet', imgpath.split('/')[-3], imgpath.split('/')[-2])
                        if not os.path.exists(savepath):
                            #print('make dirs... %s' % savepath)
                            os.makedirs(savepath)
                        img_pred = (output[b].data.permute(1, 2, 0).clamp_(0, 1).cpu().numpy()[..., ::-1] * 255).astype(numpy.uint8)
                        cv2.imwrite(os.path.join(savepath, 'im2_pred.png'), img_pred)
                        

            #batch_time.update(time.time() - t)
            #t = time.time()

            # restore the original base weight
            net.load_state_dict(weights_before)

    print("val_losses: %f" % losses.avg)
    print("val_PSNR: %f, val_SSIM: %f" % (psnrs.avg, ssims.avg))
    print("Time per batch: %.3f" % batch_time.avg)
    return psnrs.avg
# end


##########################################################
if __name__ == '__main__':
    best_PSNR = 0.0
    if cfg.mode == 'test':
        PSNR = validate(net, cfg.start_epoch)
        exit(0)

    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        train(epoch)
        
        PSNR = validate(net, epoch)
        is_best = PSNR > best_PSNR
        best_PSNR = max(PSNR, best_PSNR)
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': cfg,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_PSNR': best_PSNR
        }, is_best, cfg.exp_name)
