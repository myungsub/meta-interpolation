import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
from subprocess import call


class Middlebury(Dataset):
    def __init__(self, args): #data_root, mode='other', n_frames=2):
        '''
        :param data_root:   ./data/Middlebury
        '''
        self.args = args
        self.data_root = args.data_root
        self.mode = 'other'

        # This decides the number of frames to return
        self.nf = 4

        if self.nf == 2:
            self.image_root = os.path.join(self.data_root, self.mode + '-data-two')
        else:
            self.image_root = os.path.join(self.data_root, self.mode + '-data-all')
        self.gt_root = os.path.join(self.data_root, self.mode + '-gt-interp')
        
        self.imglist = []
        self.gt_list = []

        dir_data = sorted(glob.glob(self.image_root + '/*'))
        for _, d in enumerate(dir_data):
            _imglist = sorted(glob.glob(d + '/*.png'))
            if self.nf == 2:
                self.imglist.append(_imglist)
                self.gt_list.append(os.path.join(self.gt_root, d.split('/')[-1], 'frame10i11.png'))
            elif self.nf == 4:
                if len(_imglist) == 2:
                    continue
                elif len(_imglist) == 8:
                    _imglist = _imglist[2:6]
                    self.imglist.append(_imglist)
                    self.gt_list.append(os.path.join(self.gt_root, d.split('/')[-1], 'frame10i11.png'))
            else:
                raise ValueError('Unknown number of frames')

        self.batch_size = {'train': 1, 'val': 1, 'test': 1}
        self.current_set_name = 'val'
        self.data_length = {'train': 0, 'val': len(self.imglist), 'test': 0}

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
        imglist = self.imglist[index]
        gt_path = self.gt_list[index]

        imgs, imgpath = [], []
        for im in imglist:
            imgs.append(Image.open(im))
            imgpath.append(im)
        gt = Image.open(gt_path)

        w, h = imgs[0].size
        # if w % 32 != 0 or h % 32 != 0:
        #     w -= w % 32
        #     h -= h % 32
        #     T = transforms.Compose([
        #         transforms.Resize((h, w), interpolation=2),
        #         transforms.ToTensor()
        #     ])
        # else:
        #     T = transforms.ToTensor()
        T = transforms.ToTensor()
        for i in range(len(imgs)):
            imgs[i] = T(imgs[i])
        gt = T(gt)

        if self.args.model == 'voxelflow':  # receives 0~255 inputs
            imgs = [self.normalize(im * 255.0) for im in imgs]
            gt = self.normalize(gt * 255.0)
        elif self.args.model == 'superslomo':
            imgs = [self.normalize(im) for im in imgs]
            gt = self.normalize(gt)

        dummy_img = torch.zeros_like(gt)
        images = [imgs[0], dummy_img, imgs[1], gt, imgs[2], dummy_img, imgs[3]]
        imgpath = [imgpath[0], "", imgpath[1], gt_path, imgpath[2], "", imgpath[3]]

        metadata = {'imgpaths': imgpath}
        return images, metadata


    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name

    def __len__(self):
        return self.data_length[self.current_set_name]
