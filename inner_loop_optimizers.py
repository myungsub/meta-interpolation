import logging
import os
import math
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(GradientDescentLearningRule, self).__init__()
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = torch.ones(1) * learning_rate
        self.learning_rate.to(device)

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.9):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        updated_names_weights_dict = dict()
        for key in names_weights_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - self.learning_rate * \
                                              names_grads_wrt_params_dict[key]

        return updated_names_weights_dict


class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, optimizer, total_num_inner_loop_steps, use_learnable_learning_rates, init_learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()

        # print(init_learning_rate)
        # assert init_learning_rate > 0., 'learning_rate should be positive.'
        
        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates

        self.optimizer = optimizer
        self.state = defaultdict(dict)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.weight_decay = 0
        self.eps = 1e-8


    def initialize(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                requires_grad=self.use_learnable_learning_rates)

    def initialize_state(self):
        self.state = defaultdict(dict)
        #for idx, (key, param) in enumerate(names_weights_dict.items()):
        #    self.state[key.replace(".", "-")] = {}

    def reset(self):

        # for key, param in self.names_learning_rates_dict.items():
        #     param.fill_(self.init_learning_rate)
        pass

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        if self.optimizer == 'SGD':
            return self.update_sgd(names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1)
        elif self.optimizer == 'Adam':
            return self.update_adam(names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1)
        elif self.optimizer == 'Adamax':
            return self.update_adamax(names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1)
        else:
            raise NotImplementedError('This type of optimizer update operation is not yet implemented')

        return dict()


    def update_sgd(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1):
        """Parameter update with SGD optimizer.
        """
        updated_names_weights_dict = dict()
        for key in names_grads_wrt_params_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - \
                                              self.names_learning_rates_dict[key.replace(".", "-")][num_step] \
                                              * names_grads_wrt_params_dict[key]

        return updated_names_weights_dict


    def update_adam(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1, amsgrad=False):
        """Parameter update with Adam optimizer.
        """
        updated_names_weights_dict = dict()
        for key in names_grads_wrt_params_dict.keys():
            #names_weights_dict[key]            # param
            #names_grads_wrt_params_dict[key]   # param.grad
            if names_grads_wrt_params_dict[key] is None:
                continue
            state = self.state[key]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = self.beta1, self.beta2
            eps = self.eps

            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            if self.weight_decay != 0:
                names_grads_wrt_params_dict[key] = names_grads_wrt_params_dict[key].add(self.weight_decay, names_weights_dict[key])
            
            # Decay the first and second moment running average coefficient.
            exp_avg.mul_(beta1).add_(1 - beta1, names_grads_wrt_params_dict[key])
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, names_grads_wrt_params_dict[key], names_grads_wrt_params_dict[key])

            if amsgrad:
                # Maintain the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = self.names_learning_rates_dict[key.replace(".", "-")][num_step] / bias_correction1
            # print(key, bias_correction1, bias_correction2, step_size)

            #updated_names_weights_dict[key] = names_weights_dict[key].addcdiv(-step_size, exp_avg, denom)
            updated_names_weights_dict[key] = names_weights_dict[key].addcdiv(exp_avg, denom, value=-step_size)

        return updated_names_weights_dict


    def update_adamax(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1):
        """Parameter update with Adamax optimizer.
        """
        # print('before:')
        # print(names_weights_dict['moduleDeconv2.4.weight'][0][0].data)
        updated_names_weights_dict = dict()
        for key in names_grads_wrt_params_dict.keys():
            #names_weights_dict[key]            # param
            #names_grads_wrt_params_dict[key]   # param.grad
            if names_grads_wrt_params_dict[key] is None:
                continue
            state = self.state[key]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
                state['exp_inf'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
            exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
            beta1, beta2 = self.beta1, self.beta2
            eps = self.eps

            state['step'] += 1

            if self.weight_decay != 0:
                names_grads_wrt_params_dict[key] = names_grads_wrt_params_dict[key].add(self.weight_decay, names_weights_dict[key])
            
            # Update biased first moment estimate.
            exp_avg.mul_(beta1).add_(1 - beta1, names_grads_wrt_params_dict[key])
            # exp_avg = (beta1 * exp_avg).add(1 - beta1, names_grads_wrt_params_dict[key])
            # Update the exponentially weighted infinity norm.
            norm_buf = torch.cat([
                exp_inf.mul_(beta2).unsqueeze(0),
                names_grads_wrt_params_dict[key].abs().add_(eps).unsqueeze_(0)
            ], 0)
            #torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
            exp_inf_weighted, _ = torch.max(norm_buf, 0, keepdim=False)     # This is to remove inplace error when using Adamax optimizer
            exp_inf = exp_inf_weighted.clone()

            bias_correction = 1 - beta1 ** state['step']
            clr = self.names_learning_rates_dict[key.replace(".", "-")][num_step] / bias_correction

            updated_names_weights_dict[key] = names_weights_dict[key].addcdiv(exp_avg, exp_inf, value=-clr)

        return updated_names_weights_dict



class MetaSGDLearningRule(nn.Module):
    """Gradient descent learning rule proposed in Meta-SGD.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - task_wise_lr * dE/dp[i]
    With `task_wise_lr` as a "learnable" positive scaling parameter w.r.t. 
    each model parameters.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a SGD, Adam, or Adamax learning rule.
    """

    def __init__(self, device, optimizer, init_learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(MetaSGDLearningRule, self).__init__()

        # print(init_learning_rate)
        assert init_learning_rate > 0., 'learning_rate should be positive.'
        
        self.init_learning_rate = init_learning_rate * torch.ones(1).to(device)
        self.device = device

        self.optimizer = optimizer
        self.state = defaultdict(dict)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.weight_decay = 0
        self.eps = 1e-8


    def initialize(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                torch.ones_like(param) * self.init_learning_rate, requires_grad=True)

    def initialize_state(self):
        self.state = defaultdict(dict)
        # for idx, (key, param) in enumerate(names_weights_dict.items()):
        #    self.state[key.replace(".", "-")] = {}

    def reset(self):
        for key, param in self.names_learning_rates_dict.items():
            param.fill_(self.init_learning_rate)
        

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.1):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        if self.optimizer == 'SGD':
            return self.update_sgd(names_weights_dict, names_grads_wrt_params_dict)
        elif self.optimizer == 'Adam':
            return self.update_adam(names_weights_dict, names_grads_wrt_params_dict)
        elif self.optimizer == 'Adamax':
            return self.update_adamax(names_weights_dict, names_grads_wrt_params_dict)
        else:
            raise NotImplementedError('This type of optimizer update operation is not yet implemented')

        return dict()


    def update_sgd(self, names_weights_dict, names_grads_wrt_params_dict):
        """Parameter update with SGD optimizer.
        """
        updated_names_weights_dict = dict()
        for key in names_grads_wrt_params_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - \
                self.names_learning_rates_dict[key.replace(".", "-")] * names_grads_wrt_params_dict[key]

        return updated_names_weights_dict


    def update_adam(self, names_weights_dict, names_grads_wrt_params_dict, amsgrad=False):
        """Parameter update with Adam optimizer.
        """
        updated_names_weights_dict = dict()
        for key in names_grads_wrt_params_dict.keys():
            #names_weights_dict[key]            # param
            #names_grads_wrt_params_dict[key]   # param.grad
            if names_grads_wrt_params_dict[key] is None:
                continue
            state = self.state[key]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = self.beta1, self.beta2
            eps = self.eps

            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            if self.weight_decay != 0:
                names_grads_wrt_params_dict[key] = names_grads_wrt_params_dict[key].add(self.weight_decay, names_weights_dict[key])
            
            # Decay the first and second moment running average coefficient.
            exp_avg.mul_(beta1).add_(1 - beta1, names_grads_wrt_params_dict[key])
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, names_grads_wrt_params_dict[key], names_grads_wrt_params_dict[key])

            if amsgrad:
                # Maintain the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = self.names_learning_rates_dict[key.replace(".", "-")] / bias_correction1
            # print(key, bias_correction1, bias_correction2, step_size)

            updated_names_weights_dict[key] = names_weights_dict[key] -step_size * exp_avg / denom
            # updated_names_weights_dict[key] = names_weights_dict[key].addcdiv(exp_avg, denom, value=-step_size)

        return updated_names_weights_dict


    def update_adamax(self, names_weights_dict, names_grads_wrt_params_dict):
        """Parameter update with Adamax optimizer.
        """
        # print('before:')
        # print(names_weights_dict['moduleDeconv2.4.weight'][0][0].data)
        updated_names_weights_dict = dict()
        for key in names_grads_wrt_params_dict.keys():
            #names_weights_dict[key]            # param
            #names_grads_wrt_params_dict[key]   # param.grad
            if names_grads_wrt_params_dict[key] is None:
                continue
            state = self.state[key]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
                state['exp_inf'] = torch.zeros_like(names_weights_dict[key], memory_format=torch.preserve_format)
            exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
            beta1, beta2 = self.beta1, self.beta2
            eps = self.eps

            state['step'] += 1

            if self.weight_decay != 0:
                names_grads_wrt_params_dict[key] = names_grads_wrt_params_dict[key].add(self.weight_decay, names_weights_dict[key])
            
            # Update biased first moment estimate.
            exp_avg.mul_(beta1).add_(1 - beta1, names_grads_wrt_params_dict[key])
            # exp_avg = (beta1 * exp_avg).add(1 - beta1, names_grads_wrt_params_dict[key])
            # Update the exponentially weighted infinity norm.
            norm_buf = torch.cat([
                exp_inf.mul_(beta2).unsqueeze(0),
                names_grads_wrt_params_dict[key].abs().add_(eps).unsqueeze_(0)
            ], 0)
            #torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
            exp_inf_weighted, _ = torch.max(norm_buf, 0, keepdim=False)     # This is to remove inplace error when using Adamax optimizer
            exp_inf = exp_inf_weighted.clone()

            bias_correction = 1 - beta1 ** state['step']
            clr = self.names_learning_rates_dict[key.replace(".", "-")] / bias_correction

            # updated_names_weights_dict[key] = names_weights_dict[key].addcdiv(exp_avg, exp_inf, value=-clr)
            updated_names_weights_dict[key] = names_weights_dict[key] - clr * exp_avg / exp_inf

        return updated_names_weights_dict