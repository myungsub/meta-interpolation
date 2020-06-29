import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Video(Dataset):
    def __init__(self, args):
        self.args = args
        images = sorted(glob.glob(os.path.join(args.data_root, '*.%s' % args.img_fmt)))
        for im in images:
            try:
                float_ind = float(im.split('_')[-1][:-4])
            except ValueError:
                os.rename(im, '%s_%.06f.%s' % (im[:-4], 0.0, args.img_fmt))
        # Build batch (clip size 4)
        images = sorted(glob.glob(os.path.join(args.data_root, '*.%s' % args.img_fmt)))
        if len(images) < 4:
            print("Not enough frames for fast adaptation!")
            for _ in range(4-len(images)):
                images.append(images[-1])
            self.imglist = [images]
        else:
            self.imglist = [[images[i], images[i+1], images[i+2], images[i+3]] for i in range(len(images)-3)]
        print('[%d] images ready to be loaded' % len(self.imglist))

        self.batch_size = {'train': 0, 'val': 0, 'test': args.test_batch_size}
        self.data_length = {'train': 0, 'val': 0, 'test': len(self.imglist)}
        self.current_set_name = 'test'

        if args.model == 'superslomo':
            print('SuperSloMo normalization')
            mean = [0.429, 0.431, 0.397]
            std  = [1, 1, 1]
            self.normalize = transforms.Normalize(mean=mean, std=std)


    def __getitem__(self, index):
        imgpaths = self.imglist[index]

        # Load images
        images = [Image.open(p) for p in imgpaths]

        T = transforms.ToTensor()
        images = [T(im) for im in images]

        if self.args.model == 'superslomo':
            images = [self.normalize(im) for im in images]

        metadata = {'imgpaths': imgpaths}
        return images, metadata


    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name

    def __len__(self):
        return self.data_length[self.current_set_name]
