import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--dataset', type=str, default='vimeo90k')
data_arg.add_argument('--num_frames', type=int, default=3)
data_arg.add_argument('--data_root', type=str, default='data/vimeo_septuplet')
data_arg.add_argument('--img_fmt', type=str, default='png')

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--model', type=str, default='CAIN')
model_arg.add_argument('--depth', type=int, default=3, help='# of pooling')
model_arg.add_argument('--n_resblocks', type=int, default=12)
model_arg.add_argument('--up_mode', type=str, default='shuffle')

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'test-multi', 'gen-multi'])
learn_arg.add_argument('--loss', type=str, default='1*L1')
learn_arg.add_argument('--optimizer', type=str, default='Adam')
learn_arg.add_argument('--inner_lr', type=float, default=1e-5)
learn_arg.add_argument('--outer_lr', type=float, default=1e-5)
learn_arg.add_argument('--beta1', type=float, default=0.9)
learn_arg.add_argument('--beta2', type=float, default=0.99)
learn_arg.add_argument('--weight_decay', type=float, default=1e-4)
learn_arg.add_argument('--batch_size', type=int, default=8)
learn_arg.add_argument('--val_batch_size', type=int, default=4)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--test_mode', type=str, default='hard', help='Test mode to evaluate on SNU-FILM dataset')
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=60)
learn_arg.add_argument('--resume', action='store_true')
learn_arg.add_argument('--resume_exp', type=str, default=None)
learn_arg.add_argument('--pretrained_model', type=str, default=None)
learn_arg.add_argument('--fix_loaded', action='store_true', help='whether to fix updating all loaded parts of the model')
learn_arg.add_argument('--number_of_training_steps_per_iter', type=int, default=1)
learn_arg.add_argument('--number_of_evaluation_steps_per_iter', type=int, default=1)
learn_arg.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', action='store_true')
learn_arg.add_argument('--enable_inner_loop_optimizable_bn_params', action='store_true')
learn_arg.add_argument('--second_order', action='store_true')
learn_arg.add_argument('--first_order_to_second_order_epoch', type=int, default=-1)
learn_arg.add_argument('--use_multi_step_loss_optimization', action='store_true')
learn_arg.add_argument('--multi_step_loss_num_epochs', type=int, default=1)
learn_arg.add_argument('--total_iter_per_epoch', type=int, default=10)

# Attenuation option for L2F
learn_arg.add_argument('--attenuate', action='store_true', help='Option to switch on attenuation module')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--exp_name', type=str, default='exp')
misc_arg.add_argument('--log_iter', type=int, default=20)
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--eval_iter', type=int, default=10)
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--num_workers', type=int, default=5)
misc_arg.add_argument('--use_tensorboard', action='store_true')
misc_arg.add_argument('--viz', action='store_true', help='whether to save images')
misc_arg.add_argument('--lpips', action='store_true', help='evaluates LPIPS if set true')

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
