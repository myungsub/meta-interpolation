import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from inner_loop_optimizers import LSLRGradientDescentLearningRule
from loss import Loss
import utils


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class SceneAdaptiveInterpolation(nn.Module):
    # def __init__(self, im_shape, device, args):
    def __init__(self, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(SceneAdaptiveInterpolation, self).__init__()
        self.args = args
        self.device = torch.device('cuda') if args.cuda else torch.device('cpu')
        self.batch_size = args.batch_size
        self.use_cuda = args.cuda
        # self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.random_seed)
        # self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.num_classes_per_set,
        #                                      args=args, device=device, meta_classifier=True).to(device=self.device)
        if self.args.model == 'sepconv':
            print('Building SepConv model...')
            from sepconv.model import MetaNetwork as MetaSepConv
            self.net = MetaSepConv(resume=False if self.args.resume else True,
                                   strModel='l1').to(self.device)
        elif self.args.model == 'cain':
            print('Building CAIN model...')
            from cain.model import MetaCAIN
            self.net = MetaCAIN(depth=3, resume=False if self.args.resume else True).to(self.device)
        elif self.args.model == 'superslomo':
            print('Building SuperSloMo model...')
            from superslomo.model import MetaSuperSloMo
            self.net = MetaSuperSloMo(self.device, resume=False if self.args.resume else True).to(self.device)
            # reverse normalization to transform super-slomo outputs to 0~1 scale
            neg_mean = [-.429, -0.431, -0.397]
            std = [1, 1, 1]
            self.revNormalize = transforms.Normalize(mean=neg_mean, std=std)
        elif args.model == 'voxelflow':
            print('Building Deep VoxelFlow (DVF) model...')
            from voxelflow.core.models.voxel_flow import MetaVoxelFlow
            self.net = MetaVoxelFlow(self.args, resume=False if self.args.resume else True).to(self.device)
        else:
            raise NotImplementedError('Model not implemented yet!')

        self.inner_learning_rate = args.inner_lr
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=self.device,
                                                                    optimizer=self.args.optimizer, #'Adamax',
                                                                    init_learning_rate=self.inner_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        
        names_weights_dict = self.get_inner_loop_parameter_dict(params=self.net.named_parameters())
        self.inner_loop_optimizer.initialize(names_weights_dict=names_weights_dict)

        # Attenuator for L2F
        if self.args.attenuate:
            num_layers = len(names_weights_dict.keys())
            print('# of layers: %d' % num_layers)
            self.attenuator = nn.Sequential(
                nn.Linear(num_layers, num_layers),
                nn.ReLU(inplace=True),
                nn.Linear(num_layers, num_layers),
                nn.Sigmoid()
            ).to(device=self.device)
            # initialize to output zero
            self.gamma_mult = nn.Parameter(torch.zeros(1))

        # print("Inner Loop parameters")
        # for key, value in self.inner_loop_optimizer.named_parameters():
        #     print(key, value.shape)

        self.use_cuda = args.cuda
        self.args = args
        self.to(self.device)
        # print("Outer Loop parameters")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape, param.device, param.requires_grad)


        if self.args.optimizer == 'Adam':
            print('Using optimizer Adam.')
            if self.args.model == 'voxelflow':
                policies = self.net.get_optim_policies()
                self.optimizer = optim.Adam(policies, lr=args.outer_lr, weight_decay=args.weight_decay) #Optim(policies, args)
            else:
                self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.outer_lr, betas=(0.9, 0.99))
        elif self.args.optimizer == 'Adamax':
            print('Using optimizer Adamax.')
            self.optimizer = optim.Adamax(self.trainable_parameters(), lr=args.outer_lr, betas=(0.9, 0.999))
        else:
            self.optimizer = optim.SGD(self.trainable_parameters, lr=args.outer_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.2, patience=5, verbose=True)
        
        num_params = 0
        for param in list(self.trainable_parameters()):
            # print(param.shape)
            num_params += param.numel()
        print('# of parameters: %d' % num_params)

        self.criterion = Loss(args)

        if self.args.resume:
            print('Resume training')
            utils.load_checkpoint(args, self, None)

        if self.args.pretrained_model is not None:
            print('Loading pretrained model: %s' % self.args.pretrained_model)
            checkpoint = torch.load(self.args.pretrained_model)
            if self.args.model == 'superslomo':
                if 'state_dictFC' in checkpoint.keys():
                    utils.lossy_load_state_dict(self.net.flowComp, checkpoint['state_dictFC'])
                    utils.lossy_load_state_dict(self.net.arbTimeFlowIntrp, checkpoint['state_dictAT'])
                else:
                    utils.lossy_load_state_dict(self, checkpoint['state_dict'])
            else:
                utils.lossy_load_state_dict(self.net, checkpoint['state_dict'])


        ###### Below script needed only for distributed training
        # self.device = torch.device('cpu')
        # if torch.cuda.is_available():
        #     if torch.cuda.device_count() > 1:
        #         self.to(torch.cuda.current_device())
        #         self.classifier = nn.DataParallel(module=self.classifier)
        #     else:
        #         self.to(torch.cuda.current_device())

        #     self.device = torch.cuda.current_device()


    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights


    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    
    def get_task_embeddings(self, frames, task_id, names_weights_copy):
        support_idxs = [ [0, 2, 4], [2, 4, 6] ]  # frame indices: input[0, 2, 4, 6] --> output[3]
        # target_idx = [2, 3, 4]
        support_loss = 0
        for ind in support_idxs:
            _loss, _ = self.net_forward(frame0=frames[ind[0]][task_id].unsqueeze(0),
                                        frame1=frames[ind[2]][task_id].unsqueeze(0),
                                        target=frames[ind[1]][task_id].unsqueeze(0),
                                        weights=names_weights_copy,
                                        backup_running_statistics=True,
                                        training=True,
                                        num_step=0)
            support_loss = support_loss + _loss['total']

        self.net.zero_grad(names_weights_copy)
        grads = torch.autograd.grad(support_loss, names_weights_copy.values(),
                                    create_graph=False, allow_unused=True)

        layerwise_mean_grads = []
        for i in range(len(grads)):
            layerwise_mean_grads.append(grads[i].mean())

        layerwise_mean_grads = torch.stack(layerwise_mean_grads)

        return layerwise_mean_grads


    def attenuate_init(self, task_embeddings, names_weights_copy):
        #gamma = 0.5 + self.attenuator(task_embeddings)  # 0.5 is added to initialize gamma to 1
        gamma = 1 - self.gamma_mult * self.attenuator(task_embeddings)
        gamma.clamp_(0, 1)
        gammas = []
        for i in range(gamma.size(0)):
            gammas.append(gamma[i])
            #print(gamma[i].item())

        updated_weights = list(map(
            lambda current_params, gamma: ((gamma)*current_params.to(device=self.device)), names_weights_copy.values(), gamma))

        updated_names_weights_copy = dict(zip(names_weights_copy.keys(), updated_weights))

        return updated_names_weights_copy


    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.net.module.zero_grad(params=names_weights_copy)
        else:
            self.net.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        

        # names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        ###### This is needed for summing up gradients w.r.t. different GPUs when distributed training
        # for key, grad in names_grads_copy.items():
        #     if grad is None:
        #         print('Grads not found for inner loop parameter', key)
        #     names_grads_copy[key] = names_grads_copy[key].sum(dim=0)


        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)
        
        # loss.backward()
        # self.inner_loop_optimizer.step()

        # num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # names_weights_copy = {
        #     name.replace('module.', ''): value.unsqueeze(0).repeat(
        #         [num_devices] + [1 for i in range(len(value.shape))]) for
        #     name, value in names_weights_copy.items()}

        # names_weights_copy = {
        #     name.replace('module.', ''): value for name, value in names_weights_copy.items()}

        return names_weights_copy


    def update_loss_metrics(self, task_losses, target_loss):
        """
        :param task_losses: accumulator dictionary to gather all losses for logging to TensorBoard (updated in-place)
        :param target_loss: current loss to be updated to task_losses
        """
        for loss_key, loss_value in target_loss.items():
            if loss_key not in task_losses.keys():
                task_losses[loss_key] = utils.AverageMeter()
            task_losses[loss_key].update(loss_value.detach().cpu().data.numpy())


    def get_across_task_loss_metrics(self, total_losses, specific_losses):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        # losses['accuracy'] = np.mean(total_accuracies)
        for key, avg_meters in specific_losses.items():
            losses[key] = avg_meters.avg

        return losses


    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase, do_evaluation=False):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        frames = data_batch

        total_losses = []
        loss_accumulator = {'total': utils.AverageMeter()}
        metrics = {'psnr': utils.AverageMeter(), 'ssim': utils.AverageMeter()}
        per_task_target_preds = [[] for i in range(len(frames[0]))]
        self.net.zero_grad()
        
        for task_id in range(len(frames[0])):   # loop over batch dimension
            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()

            names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            names_weights_copy = {
                name.replace('module.', ''): value for name, value in names_weights_copy.items()}

            # inner loop
            support_idxs = [ [0, 2, 4], [2, 4, 6] ]  # frame indices: input[0, 2, 4, 6] --> output[3]
            target_idx = [2, 3, 4]
            self.inner_loop_optimizer.initialize_state()

            # Attenuate the initialization for L2F
            if self.args.attenuate:
                task_embeddings = self.get_task_embeddings(frames, task_id, names_weights_copy)
                names_weights_copy = self.attenuate_init(task_embeddings=task_embeddings,
                                                         names_weights_copy=names_weights_copy)


            for num_step in range(num_steps):
                support_loss = 0
                for ind in support_idxs:
                    _loss, _ = self.net_forward(frame0=frames[ind[0]][task_id].unsqueeze(0),
                                                frame1=frames[ind[2]][task_id].unsqueeze(0),
                                                target=frames[ind[1]][task_id].unsqueeze(0),
                                                weights=names_weights_copy,
                                                backup_running_statistics=True if (num_step == 0) else False,
                                                training=True,
                                                num_step=num_step)
                    support_loss = support_loss + _loss['total']

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                kwargs = {'backup_running_statistics': False, 'training': True, 'num_step': num_step}
                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(frame0=frames[target_idx[0]][task_id].unsqueeze(0),
                                                                 frame1=frames[target_idx[2]][task_id].unsqueeze(0),
                                                                 target=frames[target_idx[1]][task_id].unsqueeze(0),
                                                                 weights=names_weights_copy,
                                                                 **kwargs)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss['total'])
                    self.update_loss_metrics(loss_accumulator, target_loss)

                if not (use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs):
                    kwargs = {'backup_running_statistics': False, 'training': True, 'num_step': num_steps}
                    target_loss, target_preds = self.net_forward(frame0=frames[target_idx[0]][task_id].unsqueeze(0),
                                                                 frame1=frames[target_idx[2]][task_id].unsqueeze(0),
                                                                 target=frames[target_idx[1]][task_id].unsqueeze(0),
                                                                 weights=names_weights_copy,
                                                                 **kwargs)
                    task_losses.append(target_loss['total'])
                    self.update_loss_metrics(loss_accumulator, target_loss)

            per_task_target_preds[task_id] = target_preds.detach()  # target_preds.shape: (1, C, H, W)
            if do_evaluation:
                if self.args.model == 'superslomo':
                    output = self.revNormalize(target_preds.squeeze(0))
                    target = self.revNormalize(frames[target_idx[1]][task_id])
                else:
                    output = target_preds.squeeze(0)
                    target = frames[target_idx[1]][task_id]
                psnr, ssim = utils.calc_metrics(output, target)
                # print(psnr, ssim)
                metrics['psnr'].update(psnr)
                metrics['ssim'].update(ssim)
            else:
                pass

            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)

            if not training_phase:
                self.net.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   specific_losses=loss_accumulator)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds, metrics


    def net_forward(self, frame0, frame1, target, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on the input frames. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param frame0: A data batch containing the first input frames
        :param frame1: A data batch containing the second input frames
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """

        kwargs = {'backup_running_statistics': backup_running_statistics, 'num_step': num_step}
        output = self.net.forward(frame0, frame1, params=weights, **kwargs)
        # output = self.net.forward(frame0, frame1, params=None, **kwargs)
        
        if self.args.model == 'superslomo': # output becomes a tuple
            output[1]['I0'], output[1]['I1'] = frame0, frame1
            losses = self.criterion(output[0], target, **output[1])
            output = output[0]
        else:
            losses = self.criterion(output, target)

        return losses, output

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch, do_evaluation=False):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        
        losses, preds, metrics = self.forward(data_batch=data_batch, epoch=epoch,
                                              use_second_order=self.args.second_order and epoch > self.args.first_order_to_second_order_epoch,
                                              use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                              num_steps=self.args.number_of_training_steps_per_iter,
                                              training_phase=True,
                                              do_evaluation=do_evaluation)
        return losses, preds, metrics

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, preds, metrics = self.forward(data_batch=data_batch, epoch=epoch, 
                                              use_second_order=False,
                                              use_multi_step_loss_optimization=True,
                                              num_steps=self.args.number_of_evaluation_steps_per_iter,
                                              training_phase=False,
                                              do_evaluation=True)

        return losses, preds, metrics

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if False: #'imagenet' in self.args.dataset_name:
            for _, param in self.net.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        #grads = torch.autograd.grad(loss, self.net.parameters())
        #for j, param in enumerate(self.net.parameters()):
        #    param.grad = grads[j]
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch, do_evaluation=False):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        data_batch = [frame.to(device=self.device) for frame in data_batch]

        losses, preds, metrics = self.train_forward_prop(data_batch=data_batch, epoch=epoch, do_evaluation=do_evaluation)

        self.meta_update(loss=losses['loss'])
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, preds, metrics

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        data_batch = [frame.to(device=self.device) for frame in data_batch]

        losses, preds, metrics = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, preds, metrics
