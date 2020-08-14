import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SNUFILM(Dataset):
    def __init__(self, args): #data_root, mode='hard'):
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''
        self.args = args
        test_root = os.path.join(args.data_root, 'test')
        test_fn = os.path.join(args.data_root, 'test-hard-meta.txt')
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]
        
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.batch_size = {'train': 1, 'val': 1, 'test': 1}
        self.current_set_name = 'val'
        self.data_length = {'train': 0, 'val': len(self.frame_list), 'test': 0}

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
        
        print("Test dataset has %d quintuplets" %  (len(self.frame_list)))



    def __getitem__(self, index):
        
        # Use self.test_all_images:
        imgpaths = self.frame_list[index]

        images = [Image.open(p) for p in imgpaths]
        images = [self.transforms(im) for im in images]

        if self.args.model == 'voxelflow':
            images = [self.normalize(im * 255) for im in images]
        elif self.args.model == 'superslomo':
            images = [self.normalize(im) for im in images]

        imgpaths = imgpaths[:1] + [''] + imgpaths[1:4] + [''] + imgpaths[-1:]
        images = images[:1] + [torch.zeros_like(images[0])] + images[1:4] + [torch.zeros_like(images[0])] + images[-1:]
        
        metadata = {'imgpaths': imgpaths}
        return images, metadata

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name

    def __len__(self):
        return self.data_length[self.current_set_name]




