import os
import copy
import torch
import time
import argparse
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
from core import models
from core import datasets
from core.utils.optim import Optim
from core.utils.config import Config
from core.utils.eval import EvalPSNR
#from core.ops.sync_bn import DataParallelwithSyncBN

import cv2
import core.utils.transforms as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

best_PSNR = 0
VIS = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train Voxel Flow')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args


def main():
    global cfg, best_PSNR
    args = parse_args()
    cfg = Config.from_file(args.config)

    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in cfg.device)
    cudnn.benchmark = True
    cudnn.fastest = True

    #if hasattr(datasets, cfg.dataset):
    #    ds = getattr(datasets, cfg.dataset)
    #else:
    #    raise ValueError('Unknown dataset ' + cfg.dataset)

    model = getattr(models, cfg.model.name)(cfg.model).cuda()
    cfg.train.input_mean = model.input_mean
    cfg.train.input_std = model.input_std
    cfg.test.input_mean = model.input_mean
    cfg.test.input_std = model.input_std

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        #datasets.Combined240('data/combined', cfg.train),
        datasets.VimeoSeptuplet('data/vimeo_septuplet', cfg.train, True),
        # ds(cfg.train),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        # datasets.UCF101(cfg.test, False),
        datasets.VimeoSeptuplet('data/vimeo_septuplet', cfg.test, False),
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    '''test_db_list = {'ucf101': 'UCF101Voxelflow', 'middlebury': 'Middlebury'}
    test_db = 'middlebury'
    test_loader = torch.utils.data.DataLoader(
        datasets.UCF101Voxelflow('data/UCF-101', cfg.test),
        # datasets.Middlebury('data/Middlebury', cfg.test),
        #getattr(datasets, test_db_list[test_db])('data/Middlebury', cfg.test),
        batch_size=1,#cfg.test.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True)
    '''
    cfg.train.optimizer.args.max_iter = (
        cfg.train.optimizer.args.max_epoch * len(train_loader))

    policies = model.get_optim_policies()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'],
            len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = Optim(policies, cfg.train.optimizer)

    if cfg.resume or cfg.weight:
        checkpoint_path = cfg.resume if cfg.resume else cfg.weight
        if os.path.isfile(checkpoint_path):
            print(("=> loading checkpoint '{}'".format(checkpoint_path)))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'], False)
            if cfg.resume:
                optimizer.load_state_dict(checkpoint['grad_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(checkpoint_path)))

    #model = DataParallelwithSyncBN(
    #    model, device_ids=range(len(cfg.device))).cuda()
    model = model.cuda()

    # define loss function (criterion) optimizer and evaluator
    criterion = torch.nn.MSELoss().cuda()
    # evaluator = EvalPSNR(255.0 / np.mean(cfg.test.input_std))
    evaluator = EvalPSNR(255.0 / np.mean(cfg.test.input_std), cfg.test.input_mean, cfg.test.input_std)

    #PSNR = validate(val_loader, model, optimizer, criterion, evaluator)
    ##PSNR = test(test_loader, model, optimizer, criterion, evaluator)
    if cfg.mode == 'test':
        PSNR = validate(val_loader, model, optimizer, criterion, evaluator)
        return

    for epoch in range(cfg.train.optimizer.args.max_epoch):

        # train for one epoch
        train(train_loader, model, optimizer, criterion, epoch)
        # evaluate on validation set
        if ((epoch + 1) % cfg.logging.eval_freq == 0
                or epoch == cfg.train.optimizer.args.max_epoch - 1):
            PSNR = validate(val_loader, model, optimizer, criterion, evaluator)
            # remember best PSNR and save checkpoint
            is_best = PSNR > best_PSNR
            best_PSNR = max(PSNR, best_PSNR)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': dict(cfg),
                'state_dict': model.state_dict(),
                'grad_dict': optimizer.state_dict(),
                'best_PSNR': best_PSNR,
            }, is_best)


def train(train_loader, model, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    if not cfg.model.bn_training:
        print('fix batchnorm parameters')
        model.fix_batchnorm_parameters()

    end = time.time()
    for i, (imgs, _) in enumerate(train_loader):
        if i == 1000:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        lr = optimizer.adjust_learning_rate(epoch * len(train_loader) + i, epoch)

        imgs = [im.cuda() for im in imgs]


        # Meta training
        k = cfg.num_inner_update

        if True: # MAML
            base_model = copy.deepcopy(model)
            inner_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.inner_lr)

            # inner loop
            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                total_loss = 0
                for ind in indices:
                    input = torch.cat([imgs[ind[0]].clone(), imgs[ind[2]].clone()], dim=1)
                    target = imgs[ind[1]].clone()

                    input_var = torch.autograd.Variable(input)
                    target_var = torch.autograd.Variable(target)

                    # compute output
                    output = model(input_var)
                    loss = criterion(output, target_var)
                    total_loss = total_loss + loss

                inner_optimizer.zero_grad()
                total_loss.backward()
                inner_optimizer.step()

            # forward on query data
            input = torch.cat([imgs[2].clone(), imgs[4].clone()], dim=1)
            target = imgs[3].clone()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            # Copy base parameters to 'model' to connect the computation graph
            for param, base_param in zip(model.parameters(), base_model.parameters()):
                param.data = base_param.data

            # Calculate gradient & update meta-learner
            optimizer.zero_grad()
            grads = torch.autograd.grad(loss, model.parameters())
            for j, param in enumerate(model.parameters()):
                param.grad = grads[j]
            optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % cfg.logging.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                       epoch,
                       i,
                       len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       lr=lr)))
            batch_time.reset()
            data_time.reset()
            losses.reset()


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1),
               -1)[:, getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[
                           x.is_cuda])().long(), :]
    return x.view(xsize)


def validate(val_loader, model, optimizer, criterion, evaluator):
    if True: #with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        evaluator.clear()

        # switch to evaluate mode
        #model.eval()
        model.train()
        print('fix batchnorm parameters: val')
        model.fix_batchnorm_parameters()

        end = time.time()
        for i, (imgs, metadata) in enumerate(tqdm(val_loader)):
            #if i == 200:
            #    break

            imgs = [im.cuda() for im in imgs]

            weights_before = copy.deepcopy(model.state_dict())
            val_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.inner_lr)

            # inner loop
            k = cfg.num_inner_update
            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                total_loss = 0
                for ind in indices:
                    input = torch.cat([imgs[ind[0]].clone(), imgs[ind[2]].clone()], dim=1)
                    target = imgs[ind[1]].clone()
                    input_var = torch.autograd.Variable(input)
                    target_var = torch.autograd.Variable(target)

                    output = model(input_var)
                    loss = criterion(output, target_var)
                    total_loss = total_loss + loss

                val_optimizer.zero_grad()
                total_loss.backward()
                val_optimizer.step()

            input = torch.cat([imgs[2].clone(), imgs[4].clone()], dim=1)
            target = imgs[3].clone()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # restore the original base weight
            model.load_state_dict(weights_before)

            # measure accuracy and record loss
            pred = output.data.cpu().numpy()

            if VIS:
                #fig, axs = plt.subplots(1, 2)
                #print(pred[0].transpose(1, 2, 0).shape)
                #axs[0].imshow(tf.unnormalize(np.transpose(pred[0], (1, 2, 0))[..., ::-1], model.module.input_mean, model.module.input_std))
                #axs[1].imshow(tf.unnormalize(np.transpose(target[0].data.cpu().numpy(), (1, 2, 0))[..., ::-1], model.module.input_mean, model.module.input_std))
                #plt.show()
                for b in range(imgs[0].size(0)):
                    imgpath = metadata['imgpath'][b]
                    savepath = os.path.join(cfg.resume.split('.')[0], 'vimeoSeptuplet', imgpath.split('/')[-2], imgpath.split('/')[-1])
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    img_pred = tf.unnormalize(np.transpose(pred[b], (1, 2, 0))[..., ::-1], model.input_mean, model.input_std)
                    cv2.imwrite(os.path.join(savepath, 'im2_pred.png'), img_pred)
            
            evaluator(pred, target.cpu().numpy())
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % cfg.logging.print_freq == 0:

                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                       'PSNR {PSNR:.3f}'.format(
                           i,
                           len(val_loader),
                           batch_time=batch_time,
                           loss=losses,
                           PSNR=evaluator.PSNR())))

        print('Testing Results: '
              'PSNR {PSNR:.3f} ({bestPSNR:.4f})\tLoss {loss.avg:.6f}'.format(
                  PSNR=evaluator.PSNR(),
                  bestPSNR=max(evaluator.PSNR(), best_PSNR),
                  loss=losses))

        return evaluator.PSNR()


def test(test_loader, model, optimizer, criterion, evaluator):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        evaluator.clear()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target, meta) in enumerate(test_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            
            mask = meta['mask'] if ('mask' in meta) else None

            # compute output
            output = model(input_var)

            loss = criterion(output, target_var)

            # measure accuracy and record loss
            pred = output.data.cpu().numpy()
            evaluator(pred, target.cpu().numpy(), mask)
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if VIS:
                for j in range(len(output)):
                    _, axs = plt.subplots(1, 2)
                    axs[0].imshow(tf.unnormalize(np.transpose(pred[j], (1, 2, 0))[..., ::-1], model.module.input_mean, model.module.input_std))
                    axs[1].imshow(tf.unnormalize(np.transpose(target[j].cpu().numpy(), (1, 2, 0))[..., ::-1], model.module.input_mean, model.module.input_std))
                    plt.show()

            if (i + 1) % cfg.logging.print_freq == 0:

                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'PSNR {PSNR:.3f}'.format(
                           i,
                           len(test_loader),
                           batch_time=batch_time,
                           loss=losses,
                           PSNR=evaluator.PSNR())))

        print('Testing Results: '
              'PSNR {PSNR:.3f} ({bestPSNR:.4f})\tLoss {loss.avg:.5f}'.format(
                  PSNR=evaluator.PSNR(),
                  bestPSNR=max(evaluator.PSNR(), best_PSNR),
                  loss=losses))

        return evaluator.PSNR()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not cfg.output_dir:
        return
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    filename = os.path.join(cfg.output_dir, '_'.join((cfg.snapshot_pref, filename)))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(cfg.output_dir, '_'.join(
            (cfg.snapshot_pref, 'model_best.pth.tar')))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
