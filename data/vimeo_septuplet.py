import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class VimeoSeptuplet(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_root = args.data_root # 'data/vimeo_septuplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.batch_size = {'train': args.batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size}
        self.crop_size = 256
        self.frames = [1, 2, 3, 4, 5, 6, 7]

        self.current_set_name = "train" if args.mode == 'train' else 'val'

        self.data_length = {'train': len(self.trainlist), 'val': len(self.testlist), 'test': 0}

        if args.model == 'superslomo':
            print('SuperSloMo normalization')
            mean = [0.429, 0.431, 0.397]
            std  = [1, 1, 1]
            self.normalize = transforms.Normalize(mean=mean, std=std)


    def __getitem__(self, index):

        if self.current_set_name == 'train':
            imgpath = os.path.join(self.image_root, self.trainlist[index % len(self.trainlist)])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index % len(self.testlist)])

        imgpaths = ['%s/im%d.png' % (imgpath, i) for i in self.frames]

        images = [cv2.imread(p) for p in imgpaths]

        # Data augmentation
        if self.current_set_name == 'train':
            # Random crop
            H, W, C = images[0].shape
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))

            images = [v[rnd_h:rnd_h + self.crop_size,
                        rnd_w:rnd_w + self.crop_size, :] for v in images]

            # Random temporal flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]

        # BGR to RGB
        images = [im[:, :, [2, 1, 0]] for im in images]

        # Numpy to torch Tensor
        images = [torch.from_numpy(np.ascontiguousarray(np.transpose(im, (2, 0, 1)))).float() / 255 for im in images]
        metadata = {'imgpaths': imgpaths}

        if self.args.model == 'superslomo':
            images = [self.normalize(im) for im in images]

        return images, metadata

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name


    def __len__(self):
        return self.data_length[self.current_set_name]
