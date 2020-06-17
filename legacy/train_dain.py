import sys
import os
import time

import threading
import torch
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *

import cv2
import numpy
from AverageMeter import  *
from loss_function import *
import datasets
import balancedsampler
import networks
from my_args import args

import copy
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


META_ALGORITHM = args.meta # [MAML, Reptile]
TRAIN_ITER_CUT = 1e6 if args.train_iter==-1 else args.train_iter
VAL_ITER_CUT = 1e6 if args.val_iter==-1 else args.val_iter


def crop(im, maxH=640, maxW=1280):   # crop images if too big (causes out-of-memory error)
    # im.size() : NCHW
    H, W = im.size(2), im.size(3)
    return im[:, :, :min(H, maxH), :min(W, maxW)].clone()

def train():
    torch.manual_seed(args.seed)

    model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=True)
    original_model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=True)
    if args.use_cuda:
        print("Turn the model into CUDA")
        model = model.cuda()
        original_model = original_model.cuda()

    if not args.SAVED_MODEL==None:
        args.SAVED_MODEL ='./model_weights/'+ args.SAVED_MODEL + "/best" + ".pth"
        print("Fine tuning on " +  args.SAVED_MODEL)
        if not  args.use_cuda:
            pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
        else:
            pretrained_dict = torch.load(args.SAVED_MODEL)
            # model.load_state_dict(torch.load(args.SAVED_MODEL))
        #print([k for k,v in      pretrained_dict.items()])

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        # For comparison in meta training
        original_model.load_state_dict(model_dict)

        pretrained_dict = None


    if type(args.datasetName) == list:
        train_sets, test_sets = [],[]
        for ii, jj in zip(args.datasetName, args.datasetPath):
            tr_s, te_s = datasets.__dict__[ii](jj, split = args.dataset_split,single = args.single_output, task = args.task)
            train_sets.append(tr_s)
            test_sets.append(te_s)
        train_set = torch.utils.data.ConcatDataset(train_sets)
        test_set = torch.utils.data.ConcatDataset(test_sets)
    else:
        train_set, test_set = datasets.__dict__[args.datasetName](args.datasetPath)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size = args.batch_size,
        sampler=balancedsampler.RandomBalancedSampler(train_set, int(len(train_set) / args.batch_size )),
        num_workers= args.workers, pin_memory=True if args.use_cuda else False)

    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                             num_workers=args.workers, pin_memory=True if args.use_cuda else False)
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))


    # if not args.lr == 0:
    print("train the interpolation net")
    '''optimizer = torch.optim.Adamax([
                #{'params': model.initScaleNets_filter.parameters(), 'lr': args.filter_lr_coe * args.lr},
                #{'params': model.initScaleNets_filter1.parameters(), 'lr': args.filter_lr_coe * args.lr},
                #{'params': model.initScaleNets_filter2.parameters(), 'lr': args.filter_lr_coe * args.lr},
                #{'params': model.ctxNet.parameters(), 'lr': args.ctx_lr_coe * args.lr},
                #{'params': model.flownets.parameters(), 'lr': args.flow_lr_coe * args.lr},
                #{'params': model.depthNet.parameters(), 'lr': args.depth_lr_coe * args.lr},
                {'params': model.rectifyNet.parameters(), 'lr': args.rectify_lr}
                ],
                #lr=args.lr, momentum=0, weight_decay=args.weight_decay)
                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)'''
    optimizer = torch.optim.Adamax(model.rectifyNet.parameters(), lr=args.outer_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    # Fix weights for early layers
    for param in model.initScaleNets_filter.parameters():
        param.requires_grad = False
    for param in model.initScaleNets_filter1.parameters():
        param.requires_grad = False
    for param in model.initScaleNets_filter2.parameters():
        param.requires_grad = False
    for param in model.ctxNet.parameters():
        param.requires_grad = False
    for param in model.flownets.parameters():
        param.requires_grad = False
    for param in model.depthNet.parameters():
        param.requires_grad = False

    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=args.factor, patience=args.patience,verbose=True)

    print("*********Start Training********")
    print("LR is: "+ str(float(optimizer.param_groups[0]['lr'])))
    print("EPOCH is: "+ str(int(len(train_set) / args.batch_size )))
    print("Num of EPOCH is: "+ str(args.numEpoch))
    def count_network_parameters(model):

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([numpy.prod(p.size()) for p in parameters])

        return N
    print("Num. of model parameters is :" + str(count_network_parameters(model)))
    if hasattr(model,'flownets'):
        print("Num. of flow model parameters is :" +
              str(count_network_parameters(model.flownets)))
    if hasattr(model,'initScaleNets_occlusion'):
        print("Num. of initScaleNets_occlusion model parameters is :" +
              str(count_network_parameters(model.initScaleNets_occlusion) +
                  count_network_parameters(model.initScaleNets_occlusion1) +
        count_network_parameters(model.initScaleNets_occlusion2)))
    if hasattr(model,'initScaleNets_filter'):
        print("Num. of initScaleNets_filter model parameters is :" +
              str(count_network_parameters(model.initScaleNets_filter) +
                  count_network_parameters(model.initScaleNets_filter1) +
        count_network_parameters(model.initScaleNets_filter2)))
    if hasattr(model, 'ctxNet'):
        print("Num. of ctxNet model parameters is :" +
              str(count_network_parameters(model.ctxNet)))
    if hasattr(model, 'depthNet'):
        print("Num. of depthNet model parameters is :" +
              str(count_network_parameters(model.depthNet)))
    if hasattr(model,'rectifyNet'):
        print("Num. of rectifyNet model parameters is :" +
              str(count_network_parameters(model.rectifyNet)))


    training_losses = AverageMeter()
    #original_training_losses = AverageMeter()
    batch_time = AverageMeter()
    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    ikk = 0
    for kk in optimizer.param_groups:
        if kk['lr'] > 0:
            ikk = kk
            break

    
    for t in range(args.numEpoch):
        print("The id of this in-training network is " + str(args.uid))
        print(args)
        print("Learning rate for this epoch: %s" % str(round(float(ikk['lr']),7)))

        #Turn into training mode
        model = model.train()

        #for i, (X0_half,X1_half, y_half) in enumerate(train_loader):
        _t = time.time()
        for i, images in enumerate(train_loader):

            if i >= min(TRAIN_ITER_CUT, int(len(train_set) / args.batch_size )):
                #(0 if t == 0 else EPOCH):#
                break

            if args.use_cuda:
                images = [im.cuda() for im in images]
            
            images = [Variable(im, requires_grad=False) for im in images]

            # For VimeoTriplet
            #X0, y, X1 = images[0], images[1], images[2]
            # For VimeoSepTuplet
            X0, y, X1 = images[2], images[3], images[4]


            outerstepsize = args.outer_lr
            k = args.num_inner_update   # inner loop update iteration

            inner_optimizer = torch.optim.Adamax(model.rectifyNet.parameters(),
                lr=args.inner_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

            if META_ALGORITHM == "Reptile":

                # Reptile setting
                weights_before = copy.deepcopy(model.state_dict())

                for _k in range(k):
                    indices = [ [0, 2, 4], [2, 4, 6], [2, 3, 4], [0, 1, 2], [4, 5, 6] ]
                    total_loss = 0
                    for ind in indices:
                        meta_X0, meta_y, meta_X1 = images[ind[0]].clone(), images[ind[1]].clone(), images[ind[2]].clone()
                    
                        diffs, offsets, filters, occlusions = model(torch.stack((meta_X0, meta_y, meta_X1), dim=0))
                        pixel_loss, offset_loss, sym_loss = part_loss(diffs, offsets, occlusions, [meta_X0, meta_X1], epsilon=args.epsilon)
                        _total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))
                        total_loss = total_loss + _total_loss
                    # total *= 2 / len(indices)

                    inner_optimizer.zero_grad()
                    total_loss.backward()
                    inner_optimizer.step()
                
                # Reptile update
                weights_after = model.state_dict()
                model.load_state_dict({name :
                    weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                    for name in weights_before})

                with torch.no_grad():
                    diffs, offsets, filters, occlusions = model(torch.stack((X0, y, X1), dim=0))
                    pixel_loss, offset_loss, sym_loss = part_loss(diffs, offsets, occlusions, [X0, X1], epsilon=args.epsilon)
                    total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))
                training_losses.update(total_loss.item(), args.batch_size)


            elif META_ALGORITHM == "MAML":

                #weights_before = copy.deepcopy(model.state_dict())
                base_model = copy.deepcopy(model)
                #fast_weights = list(filter(lambda p: p.requires_grad, model.parameters()))

                for _k in range(k):

                    indices = [ [0, 2, 4], [2, 4, 6] ]
                    support_loss = 0
                    for ind in indices:
                        meta_X0, meta_y, meta_X1 = images[ind[0]].clone(), images[ind[1]].clone(), images[ind[2]].clone()
                    
                        diffs, offsets, filters, occlusions = model(torch.stack((meta_X0, meta_y, meta_X1), dim=0))
                        pixel_loss, offset_loss, sym_loss = part_loss(diffs, offsets, occlusions, [meta_X0, meta_X1], epsilon=args.epsilon)
                        _total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))
                        support_loss = support_loss + _total_loss
                    
                    #grad = torch.autograd.grad(loss, fast_weights)
                    #fast_weights = list(map(lambda p: p[1] - args.lr * p[0], zip(grad, fast_weights)))
                    inner_optimizer.zero_grad()
                    support_loss.backward() # create_graph=True
                    inner_optimizer.step()
                
                # Forward on query set
                diffs, offsets, filters, occlusions = model(torch.stack((X0, y, X1), dim=0))
                pixel_loss, offset_loss, sym_loss = part_loss(diffs, offsets, occlusions, [X0, X1], epsilon=args.epsilon)
                total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))
                training_losses.update(total_loss.item(), args.batch_size)

                # copy parameters to comnnect the computational graph
                for param, base_param in zip(model.rectifyNet.parameters(), base_model.rectifyNet.parameters()):
                    param.data = base_param.data

                filtered_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer.zero_grad()
                grads = torch.autograd.grad(total_loss, list(filtered_params))   # backward on weights_before: FO-MAML
                j = 0
                #print('[before update]')
                #print(list(model.parameters())[45][-1])
                for _i, param in enumerate(model.parameters()):
                    if param.requires_grad:
                        #param = param - outerstepsize * grads[j]
                        param.grad = grads[j]
                        j += 1
                optimizer.step()
                #print('[after optim.step]')
                #print(list(model.parameters())[45][-1])

            batch_time.update(time.time() - _t)
            _t = time.time()

                
            if i % 100 == 0: #max(1, int(int(len(train_set) / args.batch_size )/500.0)) == 0:

                print("Ep[%s][%05d/%d]  Time: %.2f  Pix: %s  TV: %s  Sym: %s  Total: %s  Avg. Loss: %s" % (
                      str(t), i, int(len(train_set)) // args.batch_size,
                      batch_time.avg,
                      str([round(x.item(),5) for x in pixel_loss]),
                      str([round(x.item(),4)  for x in offset_loss]),
                      str([round(x.item(), 4) for x in sym_loss]),
                      str([round(x.item(),5) for x in [total_loss]]),
                      str([round(training_losses.avg, 5)]) ))
                batch_time.reset()


        if t == 1:
            # delete the pre validation weights for cleaner workspace
            if os.path.exists(args.save_path + "/epoch" + str(0) +".pth" ):
                os.remove(args.save_path + "/epoch" + str(0) +".pth")

        if os.path.exists(args.save_path + "/epoch" + str(t-1) +".pth"):
            os.remove(args.save_path + "/epoch" + str(t-1) +".pth")
        torch.save(model.state_dict(), args.save_path + "/epoch" + str(t) +".pth")
        

        # print("\t\t**************Start Validation*****************")
        #Turn into evaluation mode

        val_total_losses = AverageMeter()
        val_total_pixel_loss = AverageMeter()
        val_total_PSNR_loss = AverageMeter()
        val_total_tv_loss = AverageMeter()
        val_total_pws_loss = AverageMeter()
        val_total_sym_loss = AverageMeter()

        for i, (images, imgpaths) in enumerate(tqdm(val_loader)):
            #if i < 50: #i < 11 or (i > 14 and i < 50):
            #    continue
            if i >=  min(VAL_ITER_CUT, int(len(test_set)/ args.batch_size)):
                break

            
            if args.use_cuda:
                images = [im.cuda() for im in images]
            #X0, y, X1 = images[0], images[1], images[2]
            #X0, y, X1 = images[2], images[3], images[4]

            # define optimizer to update the inner loop
            inner_optimizer = torch.optim.Adamax(model.rectifyNet.parameters(),
                lr=args.inner_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)


            # Reptile testing - save base model weights
            weights_base = copy.deepcopy(model.state_dict())

            k = args.num_inner_update # 2
            model.train()
            for _k in range(k):
                indices = [ [0, 2, 4], [2, 4, 6] ]
                ind = indices[_k % 2]
                meta_X0, meta_y, meta_X1 = crop(images[ind[0]]), crop(images[ind[1]]), crop(images[ind[2]])
                
                diffs, offsets, filters, occlusions, _ = model(torch.stack((meta_X0, meta_y, meta_X1), dim=0))
                pixel_loss, offset_loss, sym_loss = part_loss(diffs, offsets, occlusions, [meta_X0, meta_X1], epsilon=args.epsilon)
                total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))

                inner_optimizer.zero_grad()
                total_loss.backward()
                inner_optimizer.step()


            # Actual target validation performance
            with torch.no_grad():
                if args.datasetName == 'Vimeo_90K_sep':
                    X0, y, X1 = images[2], images[3], images[4]
                    #diffs, offsets,filters,occlusions = model(torch.stack((X0,y,X1),dim = 0))
                    diffs, offsets,filters,occlusions, output = model(torch.stack((X0,y,X1),dim = 0))

                    pixel_loss, offset_loss,sym_loss = part_loss(diffs, offsets, occlusions, [X0,X1],epsilon=args.epsilon)

                    val_total_loss = sum(x * y for x, y in zip(args.alpha, pixel_loss))

                    per_sample_pix_error = torch.mean(torch.mean(torch.mean(diffs[args.save_which] ** 2, dim=1), dim=1), dim=1)
                    per_sample_pix_error = per_sample_pix_error.data # extract tensor
                    psnr_loss = torch.mean(20 * torch.log(1.0/torch.sqrt(per_sample_pix_error)))/torch.log(torch.Tensor([10]))
                
                    val_total_losses.update(val_total_loss.item(),args.batch_size)
                    val_total_pixel_loss.update(pixel_loss[args.save_which].item(), args.batch_size)
                    val_total_tv_loss.update(offset_loss[0].item(), args.batch_size)
                    val_total_sym_loss.update(sym_loss[0].item(), args.batch_size)
                    val_total_PSNR_loss.update(psnr_loss[0],args.batch_size)

                else:       # HD_dataset testing
                    for j in range(len(images) // 2):
                        mH, mW = 720, 1280
                        X0, y, X1 = crop(images[2*j], maxH=mH, maxW=mW), crop(images[2*j+1], maxH=mH, maxW=mW), crop(images[2*j+2], maxH=mH, maxW=mW)
                        diffs, offsets,filters,occlusions , output = model(torch.stack((X0,y,X1),dim = 0))

                        pixel_loss, offset_loss,sym_loss = part_loss(diffs, offsets, occlusions, [X0,X1],epsilon=args.epsilon)

                        val_total_loss = sum(x * y for x, y in zip(args.alpha, pixel_loss))

                        per_sample_pix_error = torch.mean(torch.mean(torch.mean(diffs[args.save_which] ** 2, dim=1), dim=1), dim=1)
                        per_sample_pix_error = per_sample_pix_error.data # extract tensor
                        psnr_loss = torch.mean(20 * torch.log(1.0/torch.sqrt(per_sample_pix_error)))/torch.log(torch.Tensor([10]))
                
                        val_total_losses.update(val_total_loss.item(),args.batch_size)
                        val_total_pixel_loss.update(pixel_loss[args.save_which].item(), args.batch_size)
                        val_total_tv_loss.update(offset_loss[0].item(), args.batch_size)
                        val_total_sym_loss.update(sym_loss[0].item(), args.batch_size)
                        val_total_PSNR_loss.update(psnr_loss[0],args.batch_size)


            # Reset model to its base weights
            model.load_state_dict(weights_base)

            #del weights_base, inner_optimizer, meta_X0, meta_y, meta_X1, X0, y, X1, pixel_loss, offset_loss, sym_loss, total_loss, val_total_loss, diffs, offsets, filters, occlusions

            VIZ = False
            exp_name = 'meta_test'
            if VIZ:
                for b in range(images[0].size(0)):
                    imgpath = imgpaths[0][b]
                    savepath = os.path.join('checkpoint', exp_name, 'vimeoSeptuplet', imgpath.split('/')[-3], imgpath.split('/')[-2])
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    img_pred = (output[b].data.permute(1, 2, 0).clamp_(0, 1).cpu().numpy()[..., ::-1] * 255).astype(numpy.uint8)
                    cv2.imwrite(os.path.join(savepath, 'im2_pred.png'), img_pred)

            ''' # Original validation (not meta)
            with torch.no_grad():
                if args.use_cuda:
                    images = [im.cuda() for im in images]

                #X0, y, X1 = images[0], images[1], images[2]
                X0, y, X1 = images[2], images[3], images[4]

                    
                #diffs, offsets,filters,occlusions = model(torch.stack((X0,y,X1),dim = 0))

                pixel_loss, offset_loss,sym_loss = part_loss(diffs, offsets, occlusions, [X0,X1],epsilon=args.epsilon)

                val_total_loss = sum(x * y for x, y in zip(args.alpha, pixel_loss))

                per_sample_pix_error = torch.mean(torch.mean(torch.mean(diffs[args.save_which] ** 2,
                                                                    dim=1),dim=1),dim=1)
                per_sample_pix_error = per_sample_pix_error.data # extract tensor
                psnr_loss = torch.mean(20 * torch.log(1.0/torch.sqrt(per_sample_pix_error)))/torch.log(torch.Tensor([10]))
                #

                val_total_losses.update(val_total_loss.item(),args.batch_size)
                val_total_pixel_loss.update(pixel_loss[args.save_which].item(), args.batch_size)
                val_total_tv_loss.update(offset_loss[0].item(), args.batch_size)
                val_total_sym_loss.update(sym_loss[0].item(), args.batch_size)
                val_total_PSNR_loss.update(psnr_loss[0],args.batch_size)
                print(".",end='',flush=True)
            '''

        print("\nEpoch " + str(int(t)) +
              "\tlearning rate: " + str(float(ikk['lr'])) +
              "\tAvg Training Loss: " + str(round(training_losses.avg,5)) +
              "\tValidate Loss: " + str([round(float(val_total_losses.avg), 5)]) +
              "\tValidate PSNR: " + str([round(float(val_total_PSNR_loss.avg), 5)]) +
              "\tPixel Loss: " + str([round(float(val_total_pixel_loss.avg), 5)]) +
              "\tTV Loss: " + str([round(float(val_total_tv_loss.avg), 4)]) +
              "\tPWS Loss: " + str([round(float(val_total_pws_loss.avg), 4)]) +
              "\tSym Loss: " + str([round(float(val_total_sym_loss.avg), 4)])
              )

        auxiliary_data.append([t, float(ikk['lr']),
                                   training_losses.avg, val_total_losses.avg, val_total_pixel_loss.avg,
                                   val_total_tv_loss.avg,val_total_pws_loss.avg,val_total_sym_loss.avg])

        numpy.savetxt(args.log, numpy.array(auxiliary_data), fmt='%.8f', delimiter=',')
        training_losses.reset()
        #original_training_losses.reset()

        print("\t\tFinished an epoch, Check and Save the model weights")
            # we check the validation loss instead of training loss. OK~
        if saved_total_loss >= val_total_losses.avg:
            saved_total_loss = val_total_losses.avg
            torch.save(model.state_dict(), args.save_path + "/best"+".pth")
            print("\t\tBest Weights updated for decreased validation loss\n")

        else:
            print("\t\tWeights Not updated for undecreased validation loss\n")

        #schdule the learning rate
        scheduler.step(val_total_losses.avg)


    print("*********Finish Training********")

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=train)
    thread.start()
    thread.join()

    exit(0)
