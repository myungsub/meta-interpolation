import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
from subprocess import call


def normalize(x):
    T = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return T(x.clone())


class Middlebury(Dataset):
    def __init__(self, data_root, mode='other', n_frames=2):
        '''
        :param data_root:   ./data/Middlebury
        '''
        self.data_root = data_root
        self.mode = mode

        # This decides the number of frames to return
        self.nf = n_frames

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
            # if len(_imglist) >= 8:
            #     while len(_imglist) > 8:
            #         _imglist = _imglist[1:-1]
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
        
        # self.gt_list = []
        # dir_data = sorted(glob.glob(self.gt_root + '/*'))
        # for _, d in enumerate(dir_data):
        #     self.gt_list.append(d + '/frame10i11.png')
        #print(self.imglist, self.gt_list)


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

        meta = {'imgpath': imgpath, 'gt_path': gt_path}
        return imgs, gt, meta

    def __len__(self):
        return len(self.imglist)


def get_loader(mode, batch_size, n_frames, shuffle, num_workers):
    dataset = Middlebury('data/Middlebury', n_frames)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
