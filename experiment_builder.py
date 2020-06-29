from tqdm.auto import tqdm
import os
import numpy as np
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import utils


class ExperimentBuilder(object):
    def __init__(self, args, data, model):
        """
        Initializes an experiment builder using a named tuple (args), a data provider (data), a meta learning system
        (model) and a device (e.g. gpu/cpu/n)
        :param args: A namedtuple containing all experiment hyperparameters
        :param data: A data provider of instance MetaLearningSystemDataLoader
        :param model: A meta learning system instance
        :param device: Device/s to use for the experiment
        """
        self.args = args
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')

        # Tensorboard setup
        if self.args.mode != 'test':
            self.writer = SummaryWriter('logs/%s' % self.args.exp_name)

        self.model = model

        self.total_losses = dict()
        self.state = dict()
        self.state['best_val_loss'] = 0.
        self.state['best_val_iter'] = 0
        self.state['current_iter'] = 0
        self.start_epoch = 0
        self.best_PSNR = 0

        self.data = data(args=args, current_iter=self.state['current_iter'])

        self.state['best_epoch'] = int(self.state['best_val_iter'] / self.args.total_iter_per_epoch)
        self.epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0

        if self.args.resume:
            self.epoch = args.start_epoch
            self.state['current_iter'] = self.epoch * self.args.total_iter_per_epoch
        print(self.state['current_iter'], int(self.args.total_iter_per_epoch * self.args.max_epoch))


    def build_loss_summary_string(self, summary_losses, metrics):
        """
        Builds a progress bar summary string given current summary losses dictionary
        :param summary_losses: Current summary statistics
        :return: A summary string ready to be shown to humans.
        """
        output_update = ""
        for key, value in summary_losses.items():
            if key != 'loss' or key.find('loss_importance_vector') >= 0:
                output_update += "{}: {:.4f}, ".format(key, value)
        for key, value in metrics.items():
            output_update += "{}: {:.4f}, ".format(key, value.avg)

        return output_update


    def train_iteration(self, train_sample, sample_idx, epoch_idx, total_losses, current_iter, pbar_train, do_evaluation=False):
        """
        Runs a training iteration, updates the progress bar and returns the total and current epoch train losses.
        :param train_sample: A sample from the data provider
        :param sample_idx: The index of the incoming sample, in relation to the current training run.
        :param epoch_idx: The epoch index.
        :param total_losses: The current total losses dictionary to be updated.
        :param current_iter: The current training iteration in relation to the whole experiment.
        :param pbar_train: The progress bar of the training.
        :return: Updates total_losses, train_losses, current_iter
        """
        images, metadata = train_sample
        data_batch = images

        losses, outputs, metrics = self.model.run_train_iter(data_batch=data_batch, epoch=epoch_idx, do_evaluation=do_evaluation)

        train_output_update = self.build_loss_summary_string(losses, metrics)

        pbar_train.update(1)
        pbar_train.set_description("training phase {} -> {}".format(self.epoch, train_output_update))

        current_iter += 1

        return losses, outputs, metrics, current_iter


    def evaluation_iteration(self, val_sample, total_losses, pbar_val, phase):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_val: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        images, metadata = val_sample

        def _eval_iter(frames):
            H, W = frames[0].shape[-2:]
            if H * W > 5e5:
                if H > W:
                    images_0 = [im[:, :, :H//2, :] for im in frames]
                    images_1 = [im[:, :, H//2:, :] for im in frames]
                else:
                    images_0 = [im[:, :, :, :W//2] for im in frames]
                    images_1 = [im[:, :, :, W//2:] for im in frames]
                losses_0, outputs_0, metrics_0 = _eval_iter(images_0)
                losses_1, outputs_1, metrics_1 = _eval_iter(images_1)
                outputs = [torch.cat([outputs_0[i], outputs_1[i]], dim=2 if H > W else 3) for i in range(len(outputs_0))]
                losses = losses_0
                for k, v in losses_1.items():
                    losses[k] = (v + losses[k]) / 2
                    if k == 'loss':
                        losses[k] = losses[k].detach()
                metrics = metrics_0
                for k, v in metrics_1.items():
                    metrics[k].update(val=v.avg, n=v.count)
                del losses_0, losses_1, outputs_0, outputs_1, metrics_0, metrics_1
            else:
                losses, outputs, metrics = self.model.run_validation_iter(data_batch=frames)
                losses['loss'] = losses['loss'].detach()
            return losses, outputs, metrics

        losses, outputs, metrics = _eval_iter(images)

        val_output_update = self.build_loss_summary_string(losses, metrics)

        pbar_val.update(1)
        pbar_val.set_description("val_phase {} -> {}".format(self.epoch, val_output_update))

        return losses, outputs, metrics


    def test_iteration(self, test_sample, pbar_test, phase):
        """
        Runs a test iteration, updates the progress bar and returns the outputs.
        :param test_sample: A sample from the data provider
        :param pbar_test: The progress bar of the test stage.
        :return: The output interpolation
        """
        images, _ = test_sample
        H, W = images[0].shape[-2:]
        if H * W > 5e5:
            if H > W:
                images_0 = [im[:, :, :H//2, :] for im in images]
                images_1 = [im[:, :, H//2:, :] for im in images]
            else:
                images_0 = [im[:, :, :, :W//2] for im in images]
                images_1 = [im[:, :, :, W//2:] for im in images]
            outputs_0 = self.model.run_test_iter(data_batch=images_0)
            outputs_1 = self.model.run_test_iter(data_batch=images_1)
            outputs = [torch.cat([outputs_0[i], outputs_1[i]], dim=1 if H > W else 2) for i in range(len(outputs_0))]
        else:
            outputs = self.model.run_test_iter(data_batch=images)

        pbar_test.update(1)
        # pbar_test.set_description("test_phase -> {}".format(test_output_update))

        return outputs


    def evaluate_middlebury(self):
        """
        Runs evaluation on Middlebury dataset
        """
        return 0


    def run_experiment(self):
        """
        Runs a full training experiment with evaluations of the model on the val set at every epoch.
        """
        if self.args.mode == 'test':
            print('Start testing')
            num_test_tasks = self.data.dataset.data_length['test']
            with tqdm(total=int(num_test_tasks / self.args.test_batch_size)) as pbar_test:
                for _, test_sample in enumerate(self.data.get_test_batches(total_batches=int(num_test_tasks / self.args.test_batch_size))):
                    outputs = self.test_iteration(test_sample=test_sample,
                                                  pbar_test=pbar_test,
                                                  phase='test')
                    batch_size = test_sample[0][0].shape[0]
                    
                    for k in range(batch_size):
                        imgpath1 = test_sample[1]['imgpaths'][1][k]
                        imgpath2 = test_sample[1]['imgpaths'][2][k]
                        filename1 = imgpath1.split('/')[-1]
                        filename2 = imgpath2.split('/')[-1]
                        float_ind1 = float(filename1.split('_')[-1][:-4])
                        float_ind2 = float(filename2.split('_')[-1][:-4])
                        if float_ind2 == 0:
                            float_ind2 = 1.0
                        im_path = os.path.join(self.args.data_root, '%s_%.06f.%s' 
                            % (filename1.split('_')[0], (float_ind1 + float_ind2) / 2, self.args.img_fmt))

                        utils.save_image(outputs[k], im_path)

            print('Test finished.')
            return

        elif self.args.mode == 'val':
            print('Validation only')
            total_losses = dict()
            val_losses = dict()
            metrics_accumulator = {'psnr': utils.AverageMeter(), 'ssim': utils.AverageMeter()}
            num_evaluation_tasks = self.data.dataset.data_length['val']
            with tqdm(total=int(num_evaluation_tasks / self.args.val_batch_size)) as pbar_val:
                for _, val_sample in enumerate(self.data.get_val_batches(total_batches=int(num_evaluation_tasks / self.args.val_batch_size))):
                    val_losses, outputs, metrics = self.evaluation_iteration(val_sample=val_sample,
                                                                             total_losses=total_losses,
                                                                             pbar_val=pbar_val,
                                                                             phase='val')
                    batch_size = val_sample[0][0].shape[0]
                    for k, v in metrics.items():
                        metrics_accumulator[k].update(v.avg, n=v.count)

                    for k in range(batch_size):
                        paths = val_sample[1]['imgpaths'][3][k].split('/')
                        save_dir = os.path.join('checkpoint', self.args.exp_name, self.args.dataset, paths[-3], paths[-2])
                        if not os.path.exists(save_dir):
                            utils.makedirs(save_dir)
                        im_path = os.path.join(save_dir, paths[-1]) # 'im4.png' for VimeoSeptuplet

                        utils.save_image(outputs[0][k], im_path)
                    del val_losses, outputs, metrics

            print("%d examples processed" % metrics_accumulator['psnr'].count)
            print("PSNR: %.2f,  SSIM: %.4f\n" % (metrics_accumulator['psnr'].avg, metrics_accumulator['ssim'].avg))
            return


        with tqdm(initial=self.state['current_iter'],
                  total=int(self.args.total_iter_per_epoch * self.args.max_epoch)) as pbar_train:

            # training main loop
            while (self.state['current_iter'] < (self.args.max_epoch * self.args.total_iter_per_epoch)):

                for train_sample_idx, train_sample in enumerate(self.data.get_train_batches(total_batches=int(
                        self.args.total_iter_per_epoch * self.args.max_epoch) - self.state['current_iter'])):

                    train_losses, outputs, metrics, self.state['current_iter'] = self.train_iteration(
                        train_sample=train_sample,
                        total_losses=self.total_losses,
                        epoch_idx=(self.state['current_iter'] / self.args.total_iter_per_epoch),
                        pbar_train=pbar_train,
                        current_iter=self.state['current_iter'],
                        sample_idx=self.state['current_iter'],
                        do_evaluation=(self.state['current_iter'] % self.args.eval_iter == 0))

                    # Log to Tensorboard
                    if self.state['current_iter'] % self.args.log_iter == 1:
                        utils.log_tensorboard(self.writer, train_losses, metrics['psnr'].avg, metrics['ssim'].avg, None,
                            self.model.optimizer.param_groups[0]['lr'], self.state['current_iter'], mode='train')


                    # validation
                    if self.state['current_iter'] % self.args.total_iter_per_epoch == 0:

                        total_losses = dict()
                        val_losses = {}
                        metrics_accumulator = {'psnr': utils.AverageMeter(), 'ssim': utils.AverageMeter()}
                        num_evaluation_tasks = self.data.dataset.data_length['val']
                        with tqdm(total=int(num_evaluation_tasks / self.args.val_batch_size + 0.99)) as pbar_val:
                            for _, val_sample in enumerate(self.data.get_val_batches(total_batches=int(
                                    num_evaluation_tasks / self.args.val_batch_size + 0.99))):
                                val_loss, outputs, metrics = self.evaluation_iteration(val_sample=val_sample,
                                                                                       total_losses=total_losses,
                                                                                       pbar_val=pbar_val,
                                                                                       phase='val')
                                for k, v in metrics.items():
                                    metrics_accumulator[k].update(v.avg, n=v.count)
                                for k, v in val_loss.items():
                                    if k not in val_losses.keys():
                                        val_losses[k] = utils.AverageMeter()
                                    val_losses[k].update(v)

                                del val_loss, outputs, metrics

                            for k, v in val_losses.items():
                                val_losses[k] = v.avg
                            if val_losses["total"] < self.state['best_val_loss']:
                                print("Best validation loss", val_losses["total"])
                                self.state['best_val_loss'] = val_losses["total"]
                                self.state['best_val_psnr'] = self.state['current_iter']
                                self.state['best_epoch'] = int(self.state['best_val_iter'] / self.args.total_iter_per_epoch)
                        print("validation PSNR: %.2f,  SSIM: %.4f\n" % (metrics_accumulator['psnr'].avg, metrics_accumulator['ssim'].avg))

                        # log to TensorBoard
                        utils.log_tensorboard(self.writer, val_losses, metrics_accumulator['psnr'].avg, metrics_accumulator['ssim'].avg, None,
                            self.model.optimizer.param_groups[0]['lr'], self.state['current_iter'], mode='val')

                        self.epoch += 1

                        PSNR = metrics_accumulator['psnr'].avg
                        is_best = PSNR > self.best_PSNR
                        self.best_PSNR = max(PSNR, self.best_PSNR)
                        utils.save_checkpoint({
                            'epoch': self.epoch,
                            'arch': self.args,
                            'state_dict': self.model.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            'best_PSNR': self.best_PSNR
                        }, is_best, self.args.exp_name)

                        self.model.scheduler.step(val_losses['total'])
                        self.total_losses = dict()
                        self.epochs_done_in_this_run += 1

            #self.evaluate_middlebury()