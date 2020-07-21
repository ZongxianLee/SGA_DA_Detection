# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths


import torch
from torch.autograd import Variable
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss, mmd, minmaxscaler, new_minmaxscaler

from model.utils.parser_func import parse_args, set_dataset_args
import copy
import torch.nn.functional as F
init_lambda = 0.0053
alpha = []

if __name__ == '__main__':
    feat_1 = []
    feat_2 = []
    feat_3 = []
    mmd_avg_value = []
    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=8)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=8)
    
    data_size = len(roidb)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16_global_local import vgg16
    from model.faster_rcnn.resnet_auto import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc,
                           gc=args.gc)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))
    
    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)
    MMD = mmd

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")
    count_iter = 0
    learning_iter = 0
    v = 0
    error = []
    feat3_avg = []
    error_count = 0 
    mmd_feat3_record = []
    for epoch in range(args.start_epoch, 200):
        learning_iter_per_epoch = 0 
	save_count = 0.6
    	mmd_record = []
	error_count = 0 
        mmd_1 = torch.Tensor(10000 + 10).cuda()
        mmd_2 = torch.Tensor(10000 + 10).cuda()
        mmd_3 = torch.Tensor(10000 + 10).cuda()


    	if epoch == 1:
            thresh = init_lambda

        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()
        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)

        for step in range(10000):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            #eta = 1.0
            count_iter += 1
            #put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
            domain_p_feat1_s, domain_p_feat2_s, domain_p_feat3_s, \
            mmd_feat1_s, mmd_feat2_s, mmd_feat3_s, mmd_base1_s, mmd_base2_s, mmd_base3_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # source domain label
            domain_s = Variable(torch.zeros(domain_p_feat3_s.size(0)).long().cuda())


            #put target data into variable
            im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
            #gt is empty
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()
            domain_p_feat1_t, domain_p_feat2_t, domain_p_feat3_t, \
            mmd_feat1_t, mmd_feat2_t, mmd_feat3_t, mmd_base1_t, mmd_base2_t, mmd_base3_t = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)


            # target domain label
            domain_t = Variable(torch.ones(domain_p_feat3_t.size(0)).long().cuda())

            # domain auto learning
            # reshape
            mmd_feat1_s = torch.reshape(mmd_feat1_s,(1,-1,))
            mmd_feat2_s = torch.reshape(mmd_feat2_s,(1,-1,))
            mmd_feat3_s = torch.reshape(mmd_feat3_s,(1,-1,))

            mmd_feat1_t = torch.reshape(mmd_feat1_t,(1,-1,))
            mmd_feat2_t = torch.reshape(mmd_feat2_t,(1,-1,))
            mmd_feat3_t = torch.reshape(mmd_feat3_t,(1,-1,))
	    

            mmd_base1_s = torch.reshape(mmd_base1_s,(1,-1,))
            mmd_base2_s = torch.reshape(mmd_base2_s,(1,-1,))
            mmd_base3_s = torch.reshape(mmd_base3_s,(1,-1,))

            mmd_base1_t = torch.reshape(mmd_base1_t,(1,-1,))
            mmd_base2_t = torch.reshape(mmd_base2_t,(1,-1,))
            mmd_base3_t = torch.reshape(mmd_base3_t,(1,-1,))
	    
	   # calculate mmd for ori feature
	    mmd_feat1_ori = MMD(mmd_base1_s, mmd_base1_t)
	    mmd_feat2_ori = MMD(mmd_base2_s, mmd_base2_t)
	    mmd_feat3_ori = MMD(mmd_base3_s, mmd_base3_t)
           

            mmd_1[step] = mmd_feat1_ori.data
            mmd_2[step] = mmd_feat2_ori.data
            mmd_3[step] = mmd_feat2_ori.data

            mmd_1_scale = new_minmaxscaler(mmd_1[0:step+1])
            mmd_2_scale = new_minmaxscaler(mmd_2[0:step+1])
            mmd_3_scale = new_minmaxscaler(mmd_3[0:step+1])

	    mmd_ori_avg = (mmd_feat1_ori + mmd_feat2_ori + mmd_feat3_ori) / 3.0
   	    #mmd_record.append(copy.deepcopy(mmd_ori_avg.data * 1000))

	    if step % args.disp_interval == 0:
                if args.use_tfboard:
                    info = {
                        'mmd_feat1_ori': mmd_feat1_ori,
                        'mmd_feat2_ori': mmd_feat2_ori,
                        'mmd_feat3_ori': mmd_feat3_ori,
                        'mmd_feat_avg_ori': mmd_ori_avg
                    }
                    logger.add_scalars("logs_s_{}/mmd_ori".format(args.session), info,
                                      (epoch - 1) * iters_per_epoch + step)

####################### Computed D feature is only for observing in our experiments ###################################
            # calculate mmd for D feature
            mmd_feat1 = MMD(mmd_feat1_s, mmd_feat1_t)
            mmd_feat2 = MMD(mmd_feat2_s, mmd_feat2_t)
            mmd_feat3 = MMD(mmd_feat3_s, mmd_feat3_t)
	    mmd_avg = (mmd_feat1 + mmd_feat2 + mmd_feat3) / 3.0
            if step % args.disp_interval == 0:
                if args.use_tfboard:
                    info = {
                        'mmd_feat1': mmd_feat1,
                        'mmd_feat2': mmd_feat2,
                        'mmd_feat3': mmd_feat3,
			            'mmd_feat_avg': mmd_avg
                    }
                    logger.add_scalars("logs_s_{}/mmd".format(args.session), info,
                                      (epoch - 1) * iters_per_epoch + step)
#######################################################################################################################

	   
	
	    mmd_feat1_ori, mmd_feat2_ori, mmd_feat3_ori = minmaxscaler(mmd_feat1_ori, mmd_feat2_ori, mmd_feat3_ori)
	    mmd_feat_avg = (mmd_feat1_ori + mmd_feat2_ori + mmd_feat3_ori) / 3.0


	    mmd_feat1_detach = mmd_feat1_ori
	    mmd_feat2_detach = mmd_feat2_ori
	    mmd_feat3_detach = mmd_feat3_ori
            
	    FL1 = FocalLoss(class_num=2, gamma = mmd_feat1_detach.detach())
            FL2 = FocalLoss(class_num=2, gamma = mmd_feat2_detach.detach())
            FL3 = FocalLoss(class_num=2, gamma = mmd_feat3_detach.detach())
            
            # calculate focal loss
            dloss_s_feat1 = 0.5 * FL1(domain_p_feat1_s, domain_s)
            dloss_s_feat2 = 0.5 * FL2(domain_p_feat2_s, domain_s)
            dloss_s_feat3 = 0.5 * FL3(domain_p_feat3_s, domain_s)
	    
            dloss_t_feat1 = 0.5 * FL1(domain_p_feat1_t, domain_t)
            dloss_t_feat2 = 0.5 * FL2(domain_p_feat2_t, domain_t)
            dloss_t_feat3 = 0.5 * FL3(domain_p_feat3_t, domain_t)
	   

            loss  += ( dloss_s_feat1 + dloss_s_feat2 + dloss_s_feat3 + dloss_t_feat1 + dloss_t_feat2 + dloss_t_feat3 +  mmd_ori_avg * 0.25)
	    mmd_record.append(copy.deepcopy(mmd_ori_avg.data))
            #progressive optimization (prefer to select easy sample at the beganing of the training)
            if mmd_ori_avg  <= thresh:
		mmd_feat3_record.append(copy.deepcopy(mmd_feat3_ori.data))
            	v = 1
                loss_total  = v * loss + thresh *(0.5 * v * v - v) 
                optimizer.zero_grad()
                learning_iter += 1
		learning_iter_per_epoch += 1
		# Adjust Learning rate
                if learning_iter % 60000 == 0 and learning_iter != 0:
                    adjust_learning_rate(optimizer, args.lr_decay_gamma)
                    lr *= args.lr_decay_gamma
                    print('Adjust Learning rate')
                loss_total.backward()
                optimizer.step()
                if learning_iter % 10000 == 0 and learning_iter != 0:
		    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		    print("10000 iterations finished, save model")
		    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		    sum_feat3_score = float(sum(mmd_feat3_record)) / float(len(mmd_feat3_record))
		    feat3_avg.append(copy.deepcopy(sum_feat3_score))
		    print(feat3_avg)
		    mmd_feat3_record = []
		    save_count = save_count + 0.15
                    save_name = os.path.join(output_dir,
                                             '{}_step_{}_self_paced.pth'.format(
                                             args.dataset_t,learning_iter))
                    save_checkpoint({
                        'session': args.session,
                        'epoch': epoch + 1,
                        'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'pooling_mode': cfg.POOLING_MODE,
                        'class_agnostic': args.class_agnostic,
                    }, save_name)
                    print('save model: {}'.format(save_name))
 
            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()


                    dloss_s_feat1 = dloss_s_feat1.item()
                    dloss_t_feat1 = dloss_t_feat1.item()
                    dloss_s_feat2 = dloss_s_feat2.item()
                    dloss_t_feat2 = dloss_t_feat2.item()
                    dloss_s_feat3 = dloss_s_feat3.item()
                    dloss_t_feat3 = dloss_t_feat3.item()
		    mmd_feat_avg = mmd_ori_avg.item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f dloss_s_feat1: %.4f dloss_s_feat2: %.4f dloss_s_feat3: %.4f dloss_t_feat1: %.4f dloss_t_feat2: %.4f dloss_t_feat3: %.4f mmd_feat: %.4f eta: %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, \
                        dloss_s_feat1, dloss_s_feat2, dloss_s_feat3, \
                        dloss_t_feat1, dloss_t_feat2, dloss_t_feat3, mmd_feat_avg,\
                       args.eta))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                        'dloss_feat1': (dloss_s_feat1 + dloss_t_feat1),
                        'dloss_feat2': (dloss_s_feat2 + dloss_t_feat2),
                        'dloss_feat3': (dloss_s_feat3 + dloss_t_feat3)
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        mmd_record.sort()
	rank = int(round(0.5 * len(mmd_record)))
        thresh = mmd_record[rank]
        print_item = 'There are ' + str(learning_iter_per_epoch) + ' sample pairs are involved in training in this epoch'
        print(print_item)
    if args.use_tfboard:
        logger.close()
