import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class HD(Dataset):
    def __init__(self, args):# data_root, n_frames, is_training):
        self.args = args
        self.data_root = args.data_root #'data/HD_dataset/HD_RGB'

        self.image_root = self.data_root 

        vidlist = sorted(glob.glob(os.path.join(self.image_root, '*')))
        imglist = [sorted(glob.glob(os.path.join(v, '*.png'))) for v in vidlist]

        n_frames = 7

        imgBatch = []
        for frames in imglist:
            t = 0
            while t < len(frames):
                if len(frames) >= n_frames:
                    if t + n_frames <= len(frames):
                        imgBatch.append(frames[t:(t + n_frames)])
                    else:
                        imgBatch.append(frames[-n_frames:])
                else:
                    imgBatch.append(frames) # the whole sequence length is smaller than n_frames
                #t += n_frames
                t += 2

        self.imgBatch = imgBatch

        self.batch_size = {'train': 1, 'val': 1, 'test': 1}
        self.current_set_name = 'val'
        self.data_length = {'train': 0, 'val': len(self.imgBatch), 'test': 0}

        if args.model == 'superslomo':
            print('SuperSloMo normalization')
            mean = [0.429, 0.431, 0.397]
            std = [1, 1, 1]
            self.normalize = transforms.Normalize(mean=mean, std=std)
        elif args.model == 'voxelflow':
            print('Voxelflow normalization')
            mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
            std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
            self.normalize = transforms.Normalize(mean=mean, std=std)

        
    def __getitem__(self, index):
        imgpaths = self.imgBatch[index]

        # Load images
        imgs = [cv2.imread(p) for p in imgpaths]

        # BGR to RGB
        imgs = [im[:, :, [2, 1, 0]] for im in imgs]

        # numpy to torch tensor
        if self.args.model == 'voxelflow':
            imgs = [torch.from_numpy(np.ascontiguousarray(np.transpose(im, (2, 0, 1)))).float() for im in imgs]
        else:
            imgs = [torch.from_numpy(np.ascontiguousarray(np.transpose(im, (2, 0, 1)))).float() / 255 for im in imgs]

        if self.args.model in ['superslomo', 'voxelflow']:
            imgs = [self.normalize(im) for im in imgs]

        metadata = {'imgpaths': imgpaths}
        return imgs, metadata


    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name

    def __len__(self):
        return self.data_length[self.current_set_name]

