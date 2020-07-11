import math
import numbers
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean

def InOutPaddings(x):
    w, h = x.size(3), x.size(2)
    padding_width, padding_height = 0, 0
    if w != ((w >> 7) << 7):
        padding_width = (((w >> 7) + 1) << 7) - w
    if h != ((h >> 7) << 7):
        padding_height = (((h >> 7) + 1) << 7) - h
    paddingInput = nn.ReflectionPad2d(padding=[padding_width // 2, padding_width - padding_width // 2,
                                               padding_height // 2, padding_height - padding_height // 2])
    paddingOutput = nn.ReflectionPad2d(padding=[0 - padding_width // 2, padding_width // 2 - padding_width,
                                                0 - padding_height // 2, padding_height // 2 - padding_height])
    return paddingInput, paddingOutput


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class meanShift(nn.Module):
    def __init__(self, rgbRange, rgbMean, sign, nChannel=3):
        super(meanShift, self).__init__()
        if nChannel == 1:
            l = rgbMean[0] * rgbRange * float(sign)

            self.shifter = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.shifter.bias.data = torch.Tensor([l])
        elif nChannel == 3:  
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            self.shifter = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b])
        else:
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)
            self.shifter = nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(6).view(6, 6, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b, r, g, b])

        # Freeze the meanShift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)

        return x


def conv(in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        stride=1,
        bias=bias,
        groups=groups)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


""" CONV - (BN) - RELU - CONV - (BN) """
class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, reduction=False, bias=True, # 'reduction' is just for placeholder
                 norm=False, act=nn.ReLU(True), downscale=False):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1)
        )
        
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)

    def forward(self, x):
        res = x
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        out += res

        return out 


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)
    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)



class Interpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, 
                 reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()

        # define modules: head, body, tail
        self.headConv = conv3x3(n_feats * 2, n_feats)

        modules_body = [
            ResidualGroup(
                RCAB,
                n_resblocks=n_resblocks,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_feats, n_feats)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        return out



############################## Meta Model Utils ###############################
# from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/meta_neural_network_architectures.py



def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])
        # top_level = ".".join(name.split(".")[:-1])
        # sub_level = name.split(".")[-1]

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    # print(current_dict.keys())
    # print()
    # print(output_dict.keys())
    # print()
    return output_dict


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=True, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))
        else:
            self.bias = None

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            #print("No inner loop params")
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)

        return out

    def restore_backup_stats(self):
        pass


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            pass
            #print('no inner loop params', self)

            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            #print(num_step, params['weight'])
        else:
            #print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None


        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class MetaLayerNormLayer(nn.Module):
    def __init__(self, input_feature_shape, eps=1e-5, elementwise_affine=True):
        """
        A MetaLayerNorm layer. A layer that applies the same functionality as a layer norm layer with the added
        capability of being able to receive params at inference time to use instead of the internal ones. As well as
        being able to use its own internal weights.
        :param input_feature_shape: The input shape without the batch dimension, e.g. c, h, w
        :param eps: Epsilon to use for protection against overflows
        :param elementwise_affine: Whether to learn a multiplicative interaction parameter 'w' in addition to
        the biases.
        """
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(input_feature_shape, numbers.Integral):
            input_feature_shape = (input_feature_shape,)
        self.normalized_shape = torch.Size(input_feature_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*input_feature_shape), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor(*input_feature_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters to their initialization values.
        """
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying a layer norm function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            bias = params["bias"]
        else:
            bias = self.bias
            #print('no inner loop params', self)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MetaConvNormLayerReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaConvNormLayerReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x

        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)

        out = self.conv(out)

        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(out.shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer(out, num_step=0)

        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        conv_params = None
        activation_function_pre_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']

            conv_params = params['conv']

        out = x

        out = self.conv(out, params=conv_params)

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = F.leaky_relu(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class MetaNormLayerConvReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaNormLayerConvReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x
        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(self.input_shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer.forward(out, num_step=0)

        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)

        self.layer_dict['activation_function_pre'] = nn.LeakyReLU()

        out = self.layer_dict['activation_function_pre'].forward(self.conv.forward(out))
        print(out.shape)


    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

            conv_params = params['conv']
        else:
            conv_params = None
            #print('no inner loop params', self)

        out = x

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = self.conv.forward(out, params=conv_params)
        out = self.layer_dict['activation_function_pre'].forward(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class MetaConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(MetaConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = MetaConv2dLayer(in_feat, out_feat, stride=stride, kernel_size=kernel_size, padding=0, use_bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x, params=None):
        out = self.reflection_pad(x)

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            out = self.conv(out, params=params['conv'])
            if self.norm:
                print('no normalization used - this shouldn\'t print out')
                out = self.norm(out, params=params['norm'])
        else:
            out = self.conv(out)
            if self.norm:
                out = self.norm(out)
        return out


class MetaSequential(nn.Sequential):
    def __init__(self, *args):
        """
        A MetaSequential layer. Applies the same functionality of a standard Sequential layer with the added functionality
        of being able to receive a parameter dictionary at the forward pass which allows the forward propagation to use 
        external weights instead of the internal ones. Useful for inner loop optimization in the meta learning setting.
        """
        super(MetaSequential, self).__init__(*args)

    def is_meta_layer(self, module):
        if isinstance(module, MetaConv2dLayer) or \
           isinstance(module, MetaConvNorm) or \
           isinstance(module, MetaBatchNormLayer) or \
           isinstance(module, MetaLinearLayer) or \
           isinstance(module, MetaSequential) or \
           isinstance(module, MetaCALayer) or \
           isinstance(module, MetaRCAB) or \
           isinstance(module, MetaResidualGroup):
            return True
        else:
            return False


    def forward(self, input, params=None):
        """
        Applies a Sequential forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        for ind, module in enumerate(self):
            if params is not None and self.is_meta_layer(module):
                params = extract_top_level_dict(current_dict=params)
                input = module(input, params=params[str(ind)])
            else:
                input = module(input)
        return input

    def restore_backup_stats(self):
        pass




######################################### CAIN model utilities ###################################3
""" CONV - (BN) - RELU - CONV - (BN) """
# class MetaResBlock(nn.Module):
#     def __init__(self, in_feat, out_feat, kernel_size=3, reduction=False, bias=True, # 'reduction' is just for placeholder
#                  norm=False, act=nn.ReLU(True), downscale=False):
#         super(MetaResBlock, self).__init__()

#         self.body = MetaSequential(
#             MetaConv2dLayer(in_channels=in_feat, out_channels=out_feat, kernel_size=kernel_size, stride=2 if downscale else 1, padding=1),
#             act,
#             MetaConv2dLayer(in_channels=in_feat, out_channels=out_feat, kernel_size=kernel_size, stride=1, padding=1),
#         )
        
#         self.downscale = None
#         if downscale:
#             self.downscale = MetaConv2dLayer(in_channels=in_feat, out_channels=out_feat, kernel_size=1, stride=2, padding=0),

#     def forward(self, x, params=None):
#         res = x
#         param_dict = dict()
#         if params is not None:
#             param_dict = extract_top_level_dict(current_dict=params)
#             out = self.body(x, param_dict['body'])
#             if self.downscale is not None:
#                 res = self.downscale(res, param_dict['downscale'])
#         else:
#             out = self.body(x)
#             if self.downscale is not None:
#                 res = self.downscale(res)
#         out += res

#         return out 


## Channel Attention (CA) Layer
class MetaCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MetaCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = MetaSequential(
            MetaConv2dLayer(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False),
            MetaConv2dLayer(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, params=None):
        y = self.avg_pool(x)

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            y = self.conv_du(y, params['conv_du'])
        else:
            y = self.conv_du(y)

        return x * y, y


## Residual Channel Attention Block (RCAB)
class MetaRCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(MetaRCAB, self).__init__()

        self.body = MetaSequential(
            MetaConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            MetaConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            MetaCALayer(out_feat, reduction)
        )
        
        self.downscale = downscale
        if downscale:
            self.downscale = MetaConv2dLayer(in_channels=in_feat, out_channels=out_feat, kernel_size=3, stride=2, padding=1),
        self.return_ca = return_ca

    def forward(self, x, params=None):
        res = x
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            out, ca = self.body(x, params['body'])
            if self.downscale:
                res = self.downConv(res, params['downConv'])
        else:
            out, ca = self.body(x)
            if self.downscale:
                res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out


## Residual Group (RG)
class MetaResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(MetaResidualGroup, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(MetaConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = MetaSequential(*modules_body)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            res = self.body(x, params['body'])
        else:
            res = self.body(x)
        res += x

        return res


class MetaInterpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, 
                 reduction=16, act=nn.LeakyReLU(0.2, False), norm=False):
        super(MetaInterpolation, self).__init__()

        # define modules: head, body, tail
        self.headConv = MetaConv2dLayer(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1, padding=1)

        modules_body = [
            MetaResidualGroup(
                MetaRCAB,
                n_resblocks=n_resblocks,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(n_resgroups)]
        self.body = MetaSequential(*modules_body)

        self.tailConv = MetaConv2dLayer(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, x0, x1, params=None):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            x = self.headConv(x, param_dict['headConv'])
            res = self.body(x, param_dict['body'])
            res += x
            out = self.tailConv(res, param_dict['tailConv'])
        else:
            x = self.headConv(x)
            res = self.body(x)
            res += x
            out = self.tailConv(res)

        return out






class VGGReLUNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGReLUNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = args.num_stages
        self.num_output_classes = num_output_classes

        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)

        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)] = MetaConvNormLayerReLU(input_shape=out.shape,
                                                                        num_filters=self.cnn_filters,
                                                                        kernel_size=3, stride=self.conv_stride,
                                                                        padding=self.args.conv_padding,
                                                                        use_bias=True, args=self.args,
                                                                        normalization=True,
                                                                        meta_layer=self.meta_classifier,
                                                                        no_bn_learnable_params=False,
                                                                        device=self.device)
            out = self.layer_dict['conv{}'.format(i)](out, training=True, num_step=0)

            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)


        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        self.encoder_features_shape = list(out.shape)
        out = out.view(out.shape[0], -1)

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)
            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()
