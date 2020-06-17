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
    def __init__(self, data_root, n_frames, is_training):
        self.data_root = 'data/HD_dataset/HD_RGB'

        self.image_root = self.data_root # os.path.join(self.data_root, 'sequences')
        self.crop_size = 512

        self.training = is_training

        vidlist = sorted(glob.glob(os.path.join(self.image_root, '*')))
        imglist = [sorted(glob.glob(os.path.join(v, '*.png'))) for v in vidlist]

        self.n_frames = n_frames    # should be an odd number

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
                t += n_frames

        self.imgBatch = imgBatch

        
    def __getitem__(self, index):
        imgpaths = self.imgBatch[index]

        # Load images
        imgs = [cv2.imread(p) for p in imgpaths]

        # Data augmentation
        if False: #self.training:
            # Random crop
            H, W, C = imgs[0].shape
            rnd_h = random.randint(0, max(0, H - self.crop_size))
            rnd_w = random.randint(0, max(0, W - self.crop_size))

            imgs = [v[rnd_h:rnd_h + self.crop_size,
                      rnd_w:rnd_w + self.crop_size, :] for v in imgs]
            #gts = [v[rnd_h:rnd_h + self.crop_size,
            #         rnd_w:rnd_w + self.crop_size, :] for v in gts]

            # Random Temporal Flip
            if random.random() >= 0.5:
                imgs = imgs[::-1]
                imgpaths = imgpaths[::-1]
        #imgs = np.stack(imgs, axis=0)   # THWC

        # BGR to RGB
        imgs = [im[:, :, [2, 1, 0]] for im in imgs]

        # numpy to torch tensor
        imgs = [torch.from_numpy(np.ascontiguousarray(np.transpose(im, (2, 0, 1)))).float() / 255 for im in imgs]

        #meta = {'imgpath': imgpaths}
        return imgs, imgpaths #meta

    def __len__(self):
        return len(self.imgBatch)



def get_loader(mode, data_root, batch_size, shuffle, num_workers, n_frames=1):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = HD(data_root, is_training=is_training)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
