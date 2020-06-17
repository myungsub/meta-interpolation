from collections import defaultdict
from datetime import datetime
import os
import sys
import math
import random
import json
import glob
import logging
import shutil

import numpy as np
from scipy import signal, ndimage

import torch
from torchvision import transforms

from PIL import Image, ImageFont, ImageDraw
#from skimage.measure import compare_psnr, compare_ssim

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

import cv2

from pytorch_msssim import ssim as ssim_pth

##########################
# Training Helper Functions for making main.py clean
##########################

def load_checkpoint(args, model, optimizer, fix_loaded=False):
    if args.resume_exp is None:
        args.resume_exp = args.exp_name
    if args.mode == 'test':
        load_name = os.path.join('checkpoint', args.resume_exp, 'model_best.pth')
    else:
        load_name = os.path.join('checkpoint', args.resume_exp, 'checkpoint.pth')
    print("loading checkpoint %s" % load_name)
    checkpoint = torch.load(load_name)
    args.start_epoch = checkpoint['epoch']
    if args.resume_exp != args.exp_name:
        args.start_epoch = 0

    # filter out different keys or those with size mismatch
    model_dict = model.state_dict()
    ckpt_dict = {}
    mismatch = False
    for k, v in checkpoint['state_dict'].items():
        if k in model_dict:
            if model_dict[k].size() == v.size():
                ckpt_dict[k] = v
            else:
                print('Size mismatch while loading!   %s != %s   Skipping %s...'
                      % (str(model_dict[k].size()), str(v.size()), k))
                mismatch = True
        else:
            mismatch = True
    if len(model.state_dict().keys()) > len(ckpt_dict.keys()):
        mismatch = True
    # print(model_dict.keys())
    # print(ckpt_dict.keys())

    # Overwrite parameters to model_dict
    model_dict.update(ckpt_dict)

    # Load to model
    model.load_state_dict(model_dict)

    # if size does not match, give up on loading the optimizer.
    # if resuming from the experiment with other args.exp_name, also don't load the optimizer
    if (not mismatch) and (optimizer is not None) and (args.resume_exp is not None) and args.mode != 'test':
        optimizer.load_state_dict(checkpoint['optimizer'])
        update_lr(optimizer, args.lr)

    # if fix_loaded == True, fix the loaded model parts
    if fix_loaded:
        for k, param in model.named_parameters():
            if k in ckpt_dict.keys():
                print(k)
                param.requires_grad = False
                
    print("loaded checkpoint %s" % load_name)
    del checkpoint, ckpt_dict, model_dict


def lossy_load_state_dict(net, ckpt_state_dict, opt=None, ckpt_optimizer=None):
    model_dict = net.state_dict()
    ckpt_dict = {}
    mismatch = False
    for k, v in ckpt_state_dict.items():
        if k in model_dict:
            if model_dict[k].size() == v.size():
                ckpt_dict[k] = v
            else:
                print('Size mismatch while loading!   %s != %s   Skipping %s...'
                      % (str(model_dict[k].size()), str(v.size()), k))
                mismatch = True
    # Overwrite parameters to model_dict
    model_dict.update(ckpt_dict)
    # Load to model
    net.load_state_dict(model_dict)
    if opt is not None:
        if not mismatch:
            opt.load_state_dict(ckpt_optimizer)


def save_checkpoint(state, is_best, exp_name, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "checkpoint/%s/" % (exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint/%s/' % (exp_name) + 'model_best.pth')


# from run.py - not using it now
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
# end


##########################
# Evaluations
##########################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics


def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    lpips = AverageMeter()
    return losses, psnrs, ssims, lpips


def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


def calc_psnr(pred, gt, mask=None):
    '''
        Here we assume quantized(0-255) arguments (masked inputs if mask is not None)
    '''
    diff = (pred - gt).div(255)

    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())  # consider only masked regions
    else:
        mse = diff.pow(2).mean() + 1e-8    # mse can (surprisingly!) reach 0, which results in math domain error

    return -10 * math.log10(mse)


def calc_ssim(img1, img2, datarange=255.):
    im1 = img1.numpy().transpose(1, 2, 0).astype(np.uint8)
    im2 = img2.numpy().transpose(1, 2, 0).astype(np.uint8)
    return compare_ssim(im1, im2, datarange=datarange, multichannel=True, gaussian_weights=True)


def calc_metrics(im_pred, im_gt, mask=None):
    q_im_pred = quantize(im_pred.data, rgb_range=1.)
    q_im_gt = quantize(im_gt.data, rgb_range=1.)
    if mask is not None:
        q_im_pred = q_im_pred * mask
        q_im_gt = q_im_gt * mask
    psnr = calc_psnr(q_im_pred, q_im_gt, mask=mask)
    # ssim = calc_ssim(q_im_pred.cpu(), q_im_gt.cpu())  # This function using SciPy compare_ssim() is very, very slow
    ssim = ssim_pth(q_im_pred.unsqueeze(0), q_im_gt.unsqueeze(0), val_range=255)
    return psnr, ssim


def eval_LPIPS(model, im_pred, im_gt):
    im_pred = 2.0 * im_pred - 1
    im_gt = 2.0 * im_gt - 1
    dist = model.forward(im_pred, im_gt)[0]
    return dist


def eval_metrics(output, gt, psnrs, ssims, lpips=None, lpips_model=None, mask=None, psnrs_masked=None, ssims_masked=None):
    # PSNR should be calculated for each image
    for b in range(gt.size(0)):
        psnr, ssim = calc_metrics(output[b], gt[b], None)
        psnrs.update(psnr)
        ssims.update(ssim)
        if mask is not None:
            psnr_masked, ssim_masked = calc_metrics(output[b], gt[b], mask[b])
            psnrs_masked.update(psnr_masked)
            ssims_masked.update(ssim_masked)
        if lpips_model is not None:
            _lpips = eval_LPIPS(lpips_model, output[b].unsqueeze(0), gt[b].unsqueeze(0))
            lpips.update(_lpips)


##########################
# ETC
##########################

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def makedirs(path):
    if not os.path.exists(path):
        print("[*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        print("[*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    print("[*] {} has backup: {}".format(path, new_path))

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# TensorBoard
def log_tensorboard(writer, losses, psnr, ssim, lpips, lr, timestep, mode='train'):
    info = None
    for k, v in losses.items():
        writer.add_scalar('Loss/%s/%s' % (mode, k), v, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if lpips is not None:
        writer.add_scalar('LPIPS/%s' % mode, lpips, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)


###########################
###### VISUALIZATIONS #####
###########################

def save_image(img, path):
    # img : torch Tensor of size (C, H, W)
    q_im = quantize(img.data.mul(255))
    if len(img.size()) == 2:    # grayscale image
        im = Image.fromarray(q_im.cpu().numpy().astype(np.uint8), 'L')
    elif len(img.size()) == 3:
        im = Image.fromarray(q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
    else:
        pass
    im.save(path)

def save_batch_images(output, imgpath, save_dir, alpha=0.5):
    GEN = save_dir.find('-gen') >= 0 or save_dir.find('stereo') >= 0
    q_im_output = [quantize(o.data, rgb_range=1.) for o in output]
    for b in range(output[0].size(0)):
        paths = imgpath[0][b].split('/')
        if GEN:
            save_path = save_dir
        else:
            save_path = os.path.join(save_dir, paths[-3], paths[-2])
        makedirs(save_path)
        for o in range(len(output)):
            if o % 2 == 1 or len(output) == 1:
                output_img = Image.fromarray(q_im_output[o][b].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
                if GEN:
                    _imgname = imgpath[o//2][b].split('/')[-1]
                    imgname = "%s-%.04f.png" % (_imgname, alpha)
                else:
                    imgname = imgpath[o//2][b].split('/')[-1]

                if save_dir.find('voxelflow') >= 0:
                    #imgname = imgname.replace('gt', 'ours')
                    imgname = 'frame_01_ours.png'
                elif save_dir.find('middlebury') >= 0:
                    imgname = 'frame10i11.png'
                
                output_img.save(os.path.join(save_path, imgname))


def save_batch_images_test(output, imgpath, save_dir, alpha=0.5):
    GEN = save_dir.find('-gen') >= 0 or save_dir.find('stereo') >= 0
    q_im_output = [quantize(o.data, rgb_range=1.) for o in output]
    for b in range(output[0].size(0)):
        paths = imgpath[0][b].split('/')
        if GEN:
            save_path = save_dir
        else:
            save_path = os.path.join(save_dir, paths[-3], paths[-2])
        makedirs(save_path)
        for o in range(len(output)):
            # if o % 2 == 1 or len(output) == 1:
                # print("   ", o, b, imgpath[o][b])
                output_img = Image.fromarray(q_im_output[o][b].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
                if GEN:
                    _imgname = imgpath[o][b].split('/')[-1]
                    imgname = "%s-%.04f.png" % (_imgname, alpha)
                else:
                    imgname = imgpath[o][b].split('/')[-1]

                if save_dir.find('voxelflow') >= 0:
                    #imgname = imgname.replace('gt', 'ours')
                    imgname = 'frame_01_ours.png'
                elif save_dir.find('middlebury') >= 0:
                    imgname = 'frame10i11.png'

                output_img.save(os.path.join(save_path, imgname))


def save_images_test(output, imgpath, save_dir, alpha=0.5):
    q_im_output = [quantize(o.data, rgb_range=1.) for o in output]
    for b in range(output[0].size(0)):
        paths = imgpath[1][b].split('/')
        save_path = os.path.join(save_dir, paths[-3], paths[-2])
        makedirs(save_path)
        # Output length is one
        output_img = Image.fromarray(q_im_output[0][b].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
        imgname = imgpath[1][b].split('/')[-1]

        # if save_dir.find('voxelflow') >= 0:
        #     imgname = 'frame_01_ours.png'
        # elif save_dir.find('middlebury') >= 0:
        #     imgname = 'frame10i11.png'

        output_img.save(os.path.join(save_path, imgname))


def make_video(out_dir, gt_dir, gt_first=False):
    gt_ext = '/*.png'
    frames_all = sorted(glob.glob(out_dir + '/*.png') + glob.glob(gt_dir + gt_ext), \
        key=lambda frame: frame.split('/')[-1])
    print("# of total frames : %d" % len(frames_all))
    if gt_first:
        print("Appending GT in front..")
        frames_all = sorted(glob.glob(gt_dir + gt_ext)) + frames_all
        print("# of total frames : %d" % len(frames_all))

    # Read the first image to determine height and width
    frame = cv2.imread(frames_all[0])
    h, w, _ = frame.shape

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_dir + '/slomo.mp4', fourcc, 30, (w, h))
    for p in frames_all:
        #print(p)
        # TODO: add captions (e.g. 'GT', 'slow motion x4')
        frame = cv2.imread(p)
        fh, fw = frame.shape[:2]
        #print(fh, fw, h, w)
        if fh != h or fw != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        out.write(frame)

def check_already_extracted(vid):
    return bool(os.path.exists(vid + '/00001.png'))
