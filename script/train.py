import os
import shelve
import time
import pprint
import torch
import cv2 as cv
import numpy as np
import shutil
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from dataset.collate import collate_train
from lib.model.vgg16 import VGG16
from lib.model.resnet import Resnet
from utils.net_utils import clip_gradient
from utils.data_utils import *
from utils.debug_plots import plot_debug_info
from utils.visdom_plot import *

watch_list = ['000012', '000017', '000019', '000021', '000026', '000036', '000089', '000102', '000121', '000130', '000198']

def inverse_transform(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x = image.new(*image.size())
    x[0, :, :] = image[0, :, :] * std[0] + mean[0]
    x[1, :, :] = image[1, :, :] * std[1] + mean[1]
    x[2, :, :] = image[2, :, :] * std[2] + mean[2]
    x = (x * 255).clamp_(0, 255)

    x = x.permute(1, 2, 0).cpu().numpy()
    offset = cfg.VIS_OFFSET
    bigx = np.zeros([x.shape[0]+2*offset, x.shape[1]+2*offset, x.shape[2]], dtype=np.float)
    bigx[offset:x.shape[0]+offset, offset:x.shape[1]+offset, :] = x
    return bigx


def train(dataset_name, net, batch_size, learning_rate, optimizer, lr_decay_step,
          lr_decay_gamma, pretrain, resume, class_agnostic, total_epoch,
          display_interval, session, epoch, save_dir, vis_off, visdom_port, log, mGPU, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    print(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))
    global watch_list

    if batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = batch_size
    if learning_rate is not None:
        cfg.TRAIN.LEARNING_RATE = learning_rate
    if lr_decay_step is not None:
        cfg.TRAIN.LR_DECAY_STEP = lr_decay_step
    if lr_decay_gamma is not None:
        cfg.TRAIN.LR_DECAY_GAMMA = lr_decay_gamma

    if 'cfg_file' in add_params:
        update_config_from_file(add_params['cfg_file'])

    log.info(Back.WHITE + Fore.BLACK + 'Using config:')
    log.info('GENERAL:')
    log.info(cfg.GENERAL)
    log.info('TRAIN:')
    log.info(cfg.TRAIN)
    log.info('RPN:')
    log.info(cfg.RPN)

    dataset, ds_name = dataset_factory.get_dataset(dataset_name, add_params)
    cfg.TRAIN.DATA_SET = dataset_name.split("_")[0]
    #watch_list = watch_list + dataset._image_index[0::800]
    #dataset._image_data = dataset._image_data[:cfg.TRAIN.SUP_SAMPLES]
    #dataset._image_index = dataset._image_index[:cfg.TRAIN.SUP_SAMPLES]
    dataset, watch_ids = change_semisupervised_sample_ratio(dataset, cfg.TRAIN.SAMPLING_RATIO, 10000)
    #watch_ids = dataset._image_index
    #dataset = combined_pascal_dataset(dataset, add_params)
    #dataset,watch_ids = semisupervised_sampling_coco(dataset, 11000)
    loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True, collate_fn=collate_train)

    #if 'data_path' in add_params:
    #    cfg.DATA_DIR = add_params['data_path']
    output_dir = os.path.join(cfg.DATA_DIR, save_dir, net, ds_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(Back.CYAN + Fore.BLACK + 'Output directory: %s' % (output_dir))

    pretrained = True
    model_name = '{}.pth'.format(net)
    if 'use_pretrain' in add_params:
        pretrained = add_params['use_pretrain']
    if 'model_name' in add_params:
        model_name = '{}.pth'.format(add_params['model_name'])
    model_path = os.path.join(cfg.DATA_DIR, 'pretrained_model', model_name)
    if net == 'vgg16':
        faster_rcnn = VGG16(dataset.num_classes, class_agnostic=class_agnostic,
                            pretrained=pretrained, model_path=model_path)
    elif net.startswith('resnet'):
        num_layers = net[6:]
        faster_rcnn = Resnet(num_layers, dataset.num_classes, class_agnostic=class_agnostic,
                             pretrained=pretrained, model_path=model_path)
    else:
        raise ValueError(Back.RED + 'Network "{}" is not defined!'.format(net))

    faster_rcnn.init()
    faster_rcnn.to(device)

    params = []
    for key, value in dict(faster_rcnn.named_parameters()).items():
        #if key.split('.')[0]=='RCNN_base':
        #    continue
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value],
                            'lr': cfg.TRAIN.LEARNING_RATE * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value],
                            'lr':cfg.TRAIN.LEARNING_RATE,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if optimizer == 'sgd':
        optimizer = SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    elif optimizer == 'adam':
        optimizer = Adam(params)
    else:
        raise ValueError(Back.RED + 'Optimizer "{}" is not defined!'.format(optimizer))
    print("Optimizer initialized")
    start_epoch = 1

    if pretrain or resume:
        model_name = 'frcnn_{}_{}.pth'.format(session, epoch)
        if 'model_name' in add_params:
            model_name = '{}.pth'.format(add_params['model_name'])
        model_path = os.path.join(output_dir, model_name)
        print(Back.WHITE + Fore.BLACK + 'Loading checkpoint %s...' % (model_path))
        checkpoint = torch.load(model_path, map_location=device)
        faster_rcnn.load_state_dict(checkpoint['model'])
        if resume:
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        print('Done.')

    # Decays the learning rate of each parameter group by gamma every step_size epochs..
    #lr_scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP,
    #                      gamma=cfg.TRAIN.LR_DECAY_GAMMA)
    lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.LR_DECAY_GAMMA)

    if mGPU:
        faster_rcnn = torch.nn.DataParallel(faster_rcnn)

    faster_rcnn.train()
    print("Start training")
    if not vis_off:
        from visualize.plotter import Plotter
        plotter = Plotter()
    #visdom_plotter = VisdomLinePlotter(port=visdom_port)

    #loss fraction
    loss_fraction = cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.PROPOSAL_PER_IMG

    #selective search box scores stored as dict with image id as the key.
    data_size = len(dataset)
    sampled_boxes = {}

    #clear debug directory
    debug_dir = os.path.join(cfg.DATA_DIR, 'debug', 'session_'+str(session))
    if not resume and os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
    elif not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if cfg.TRAIN.USE_SS_GTBOXES:
        save_dir = os.path.join(debug_dir, 'ss_box_scores')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if cfg.TRAIN.DATA_SET=="voc":
            if resume:
                save_path = os.path.join(save_dir, 'score_epoch_' + str(epoch) + '.pt')
                pseudo_GT_scores = torch.load(save_path)
            else:
                pseudo_GT_scores = {}
                for i in range(len(dataset._image_index)):
                    pseudo_GT_scores[dataset._image_index[i]] = torch.zeros((cfg.TRAIN.NUM_PROPOSALS, dataset.num_classes), device=device) #pseudo gt_box classes and scores
        elif cfg.TRAIN.DATA_SET=="coco":
            save_path = os.path.join(save_dir, 'score_epoch_' + str(epoch))
            if not resume:
                score_db = shelve.open(save_path, "c")
                score_db_next_epoch = score_db

    flags = []
    for current_epoch in range(start_epoch, total_epoch + 1):
        loss_temp = 0
        start = time.time()
        total_rpn_loss_cls = 0.0
        total_rpn_loss_bbox = 0.0
        total_rcnn_loss_cls = 0.0
        total_rcnn_loss_bbox = 0.0
        total_loss = 0.0
        summary_record = torch.empty([0, 6], dtype=torch.float).to(device=device)

        if current_epoch>1:
            del add_params['devkit_path']
            dataset, ds_name = dataset_factory.get_dataset(dataset_name, add_params)
            dataset, _ = change_semisupervised_sample_ratio(dataset, cfg.TRAIN.SAMPLING_RATIO, 10000)
            #dataset = combined_pascal_dataset(dataset, add_params)
            loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, collate_fn=collate_train)
            if cfg.TRAIN.DATA_SET=="coco" and cfg.TRAIN.USE_SS_GTBOXES:
                save_path = os.path.join(save_dir, 'score_epoch_' + str(epoch-1))
                score_db = shelve.open(save_path)
                save_path = os.path.join(save_dir, 'score_epoch_' + str(epoch))
                score_db_next_epoch = shelve.open(save_path, "c")

        for step, data in enumerate(loader):
            image_data = data[0].to(device)
            image_info = data[1].to(device)
            gt_boxes = data[2].to(device)
            image_labels = data[3]
            image_ids = data[4]
            real_gt_boxes = data[5].to(device)

            #load the pseudoGT scores
            shuffle_indices = []
            if cfg.TRAIN.USE_SS_GTBOXES:
                for i, id in enumerate(image_ids):
                    num_boxes = int(image_info[i, 3])
                    try:
                        if cfg.TRAIN.DATA_SET=="voc":
                            gt_boxes[i, :num_boxes, 4:] = pseudo_GT_scores[id][:num_boxes]
                        else:
                            gt_boxes[i, :num_boxes, 4:] = score_db[id]
                    except:
                        pass
                    flags.append(True if image_ids[i] in watch_ids else False)
            *_, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, gt_boxes, debug_info = faster_rcnn(image_data, image_info, gt_boxes, image_labels, current_epoch, flags, real_gt_boxes=real_gt_boxes)
            #save the updated GT box scores
            if cfg.TRAIN.USE_SS_GTBOXES:
                #summary_record = torch.cat([summary_record, debug_info['summary_record']], dim=0)
                for i, id in enumerate(image_ids):
                    if flags[i]==False:
                        num_sampled_boxes = debug_info['num_sampled_boxes'][i]
                        sampled_boxes[id] = debug_info['sampled_gt_boxes'][i, :num_sampled_boxes] / image_info[i, 2].item()
                        num_boxes = int(image_info[i, 3])
                        if cfg.TRAIN.DATA_SET=="voc":
                            if id not in pseudo_GT_scores.keys():
                                pseudo_GT_scores[id] = torch.zeros((cfg.TRAIN.NUM_PROPOSALS, dataset.num_classes), device=device)
                            pseudo_GT_scores[id][:num_boxes] = gt_boxes[i, :num_boxes, 4:]
                        else:
                            score_db_next_epoch[id] = gt_boxes[i, :num_boxes, 4:]
                        #if id in watch_list:
                        #    image = inverse_transform(image_data[i])
                        #    image = cv.cvtColor(image.astype(np.uint8), cv.COLOR_RGB2BGR)
                        #    debug_info['gt_boxes'] = updated_gt_boxes[i]
                        #    debug_info['gt_scores'] = pseudo_GT_scores[id].cpu().numpy()
                        #    debug_info['save_root_dir'] = debug_dir
                        #    debug_info['num_boxes_this_image'] = int(debug_info['image_info'][i, 3])
                        #    plot_debug_info(image, debug_info, i, current_epoch, id, dataset.classes)
                flags = []
            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() \
                + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()
            total_rpn_loss_cls += rpn_loss_cls.mean().item() * loss_fraction
            total_rpn_loss_bbox += rpn_loss_bbox.mean().item() * loss_fraction
            total_rcnn_loss_cls += RCNN_loss_cls.mean().item() * loss_fraction
            total_rcnn_loss_bbox += RCNN_loss_bbox.mean().item() * loss_fraction
            total_loss += loss.item() * loss_fraction

            optimizer.zero_grad()
            loss.backward()
            if net == 'vgg16':
                clip_gradient(faster_rcnn, 10.)
            optimizer.step()

            if step % display_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (display_interval + 1)

                loss_rpn_cls = rpn_loss_cls.mean().item()
                loss_rpn_bbox = rpn_loss_bbox.mean().item()
                loss_rcnn_cls = RCNN_loss_cls.mean().item()
                loss_rcnn_bbox = RCNN_loss_bbox.mean().item()

                print(Back.WHITE + Fore.BLACK + '[session %d][epoch %2d/%2d][iter %4d/%4d]'
                      % (session, current_epoch, total_epoch, step, len(loader)))
                print('loss: %.4f, learning rate: %.2e, time cost: %f'
                      % (loss_temp, optimizer.param_groups[0]['lr'], end-start))
                print('rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f'
                      % (loss_rpn_cls, loss_rpn_bbox, loss_rcnn_cls, loss_rcnn_bbox))

                if not vis_off:
                    plotter_data = {'session': session,
                                    'current_epoch': current_epoch,
                                    'total_epoch': total_epoch,
                                    'current_iter': step,
                                    'total_iter': len(loader),
                                    'lr': optimizer.param_groups[0]['lr'],
                                    'time_cost': end-start,
                                    'loss': [loss_temp,
                                             loss_rpn_cls,
                                             loss_rpn_bbox,
                                             loss_rcnn_cls,
                                             loss_rcnn_bbox]}
                    plotter.send('data', plotter_data)
                loss_temp = 0
                start = time.time()

            #if step>200:
            #    break

        if cfg.TRAIN.USE_SS_GTBOXES:
            #save model scores at this epoch
            if cfg.TRAIN.DATA_SET == "voc":
                save_path = os.path.join(save_dir, 'score_epoch_' + str(current_epoch) + '.pt')
                torch.save(pseudo_GT_scores, save_path)
            else:
                score_db_next_epoch.close()

            #save sampled boxes
            #save_dir = os.path.join(debug_dir, 'sampled_boxes')
            #if not os.path.exists(save_dir):
            #    os.makedirs(save_dir)
            #save_path = os.path.join(save_dir, 'sampled_boxes_' + str(current_epoch) + '.pt')
            #torch.save(sampled_boxes, save_path)


        total_rpn_loss_cls = total_rpn_loss_cls / (data_size * cfg.TRAIN.PROPOSAL_PER_IMG)
        total_rpn_loss_bbox = total_rpn_loss_bbox / (data_size * cfg.TRAIN.PROPOSAL_PER_IMG)
        total_rcnn_loss_cls = total_rcnn_loss_cls / (data_size * cfg.TRAIN.PROPOSAL_PER_IMG)
        total_rcnn_loss_bbox = total_rcnn_loss_bbox / (data_size * cfg.TRAIN.PROPOSAL_PER_IMG)
        total_loss = total_loss / (data_size * cfg.TRAIN.PROPOSAL_PER_IMG)
        #visdom_plotter.plot("RPN_loss", "train", "RPN loss", current_epoch, total_rpn_loss_cls, "loss_cls")
        #visdom_plotter.plot("RPN_loss", "train", "RPN loss", current_epoch, total_rpn_loss_bbox, "loss_bbox")
        #visdom_plotter.plot("RCNN_loss", "train", "RCNN loss", current_epoch, total_rcnn_loss_cls, "loss_cls")
        #visdom_plotter.plot("RCNN_loss", "train", "RCNN loss", current_epoch, total_rcnn_loss_bbox, "loss_bbox")
        #visdom_plotter.plot("TOTAL_loss", "train", "Total loss", current_epoch, total_loss, "loss_total")
        log.info("Epoch: {} RPN loss_cls: {:.3f} RPN loss_bbox: {:.3f} RCNN loss_cls: {:.3f} RCNN loss_bbox: {:.3f} Total loss: {:.3f}".format(
            current_epoch, total_rpn_loss_cls, total_rpn_loss_bbox, total_rcnn_loss_cls, total_rcnn_loss_bbox, total_loss))
        lr_scheduler.step()

        save_path = os.path.join(output_dir, 'frcnn_{}_{}.pth'.format(session, current_epoch))
        checkpoint = {'epoch': current_epoch + 1,
                      'model': faster_rcnn.module().state_dict() if mGPU else faster_rcnn.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, save_path)
        if not vis_off:
            plotter.send('save', save_path[:-4])
        print(Back.WHITE + Fore.BLACK + 'Model saved: %s' % (save_path))

    if not vis_off:
        plotter.send('close', None)
    print(Back.GREEN + Fore.BLACK + 'Train finished.')
