import numpy as np

import core.utils.transforms as tf
class EvalPSNR(object):

    def __init__(self, max_level, mean, std):
        self.max_level = max_level
        ####
        self.mean = mean
        self.std = std
        ####
        self.clear()

    def __call__(self, pred, gt, mask=None):
        assert (pred.shape == gt.shape)
        
        if mask is None:
            mask = np.ones((pred.shape[0], pred.shape[2], pred.shape[3]))
        else:
            mask = mask >= 128
        for i in range(pred.shape[0]):
            temp = np.tile(mask[i, np.newaxis, :, :], (3, 1, 1))
            # temp = np.tile(mask[i, :, :, np.newaxis], (1, 1, 3))
            if np.sum(temp) == 0:
                continue
            # print(pred[i].max(), pred[i].min(), gt[i].max(), gt[i].min())
            # _pred = tf.unnormalize(np.transpose(pred[i], (1, 2, 0)), self.mean, self.std).astype(np.float32)
            # _gt = tf.unnormalize(np.transpose(gt[i], (1, 2, 0)), self.mean, self.std).astype(np.float32)
            # delta = (_pred - _gt) * temp
            
            # input()
            ####
            delta = (pred[i, :, :, :] - gt[i, :, :, :]) * temp
            delta = np.sum(np.square(delta)) / np.sum(temp)
            p = 10 * np.log10(self.max_level * self.max_level / delta)
            # print(p)
            self.psnr += p#10 * np.log10(self.max_level * self.max_level / delta)
            # self.psnr.append(10 * np.log10(self.max_level * self.max_level / delta))
            self.count += 1
            

    def PSNR(self):
        return np.sum(np.array(self.psnr)) / max(1, self.count)

    def clear(self):
        # self.psnr = []
        self.psnr = 0
        self.count = 0
