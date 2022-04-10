import os
import time
import pprint
import torch
import cv2 as cv
import numpy as np
import shutil
import torch.nn as nn
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from dataset.collate import collate_train, collate_test
from lib.model.clf_net import Cls_Net
from utils.visdom_plot import *

watch_list = ['000012', '000017', '000019', '000021', '000026', '000036', '000089', '000102', '000121', '000130', '000198']


def cam_train(dataset, net, batch_size, learning_rate, resume, total_epoch,
          display_interval, session, epoch, save_dir, visdom_port, log, mGPU, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    print(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))

    if batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = batch_size
    if learning_rate is not None:
        cfg.TRAIN.LEARNING_RATE = learning_rate

    log.info(Back.WHITE + Fore.BLACK + 'Using config:')
    log.info('GENERAL:')
    log.info(cfg.GENERAL)
    log.info('TRAIN:')
    log.info(cfg.TRAIN)

    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params)
    test_dataset, _ = dataset_factory.get_dataset("coco_2017_val", add_params, mode="test")
    loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test)
    if 'data_path' in add_params:
        cfg.DATA_DIR = add_params['data_path']
    output_dir = os.path.join(cfg.DATA_DIR, save_dir, net, ds_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(Back.CYAN + Fore.BLACK + 'Output directory: %s' % (output_dir))

    cam_model = Cls_Net(dataset.num_classes-1)
    cam_model.to(device)

    optimizer = SGD(cam_model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    print("Optimizer initialized")
    start_epoch = 1

    if resume:
        model_name = 'cam_{}_{}.pth'.format(session, epoch)
        model_path = os.path.join(output_dir, model_name)
        print(Back.WHITE + Fore.BLACK + 'Loading checkpoint %s...' % (model_path))
        checkpoint = torch.load(model_path, map_location=device)
        cam_model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Done.')

    # Decays the learning rate of each parameter group by gamma every step_size epochs..
    #lr_scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP,
    #                      gamma=cfg.TRAIN.LR_DECAY_GAMMA)
    lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.CAM_MILESTONES, gamma=cfg.TRAIN.LR_DECAY_GAMMA)

    if mGPU:
        cam_model = torch.nn.DataParallel(cam_model)

    print("Start training")
    visdom_plotter = VisdomLinePlotter(port=visdom_port)

    train_data_size = len(dataset)
    test_data_size = len(test_dataset)
    criterion = nn.BCEWithLogitsLoss()

    for current_epoch in range(start_epoch, total_epoch + 1):
        loss_temp = 0
        start = time.time()
        cam_model.train()

        for step, data in enumerate(loader):
            image_data = data[0].to(device)
            image_info = data[1].to(device)
            image_labels = data[3]
            image_ids = data[4]

            logits, targets = cam_model(image_data, image_labels)
            loss = criterion(logits, targets)
            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % display_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (display_interval + 1)

                print(Back.WHITE + Fore.BLACK + '[TRAIN session %d][epoch %2d/%2d][iter %4d/%4d]'
                      % (session, current_epoch, total_epoch, step, len(loader)))
                print('loss: %.4f, learning rate: %.2e, time cost: %f'
                      % (loss_temp, optimizer.param_groups[0]['lr'], end-start))

                loss_temp = 0
                start = time.time()

            #if step>800:
            #    break

        lr_scheduler.step()
        optimizer.zero_grad()

        all_preds = torch.empty([0, dataset.num_classes-1], dtype=torch.float).to(device=device)
        all_targets =  torch.empty([0, dataset.num_classes-1], dtype=torch.float).to(device=device)
        cam_model.eval()
        
        for step, data in enumerate(test_loader):
            image_data = data[0].to(device)
            image_info = data[1].to(device)
            image_labels = data[3]
            image_ids = data[4]

            with torch.no_grad():
                logits, targets = cam_model(image_data, image_labels)
                outputs = torch.sigmoid(logits)
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                all_preds = torch.cat([all_preds, outputs], dim=0)
                all_targets = torch.cat([all_targets, targets], dim=0)
            if step % display_interval == 0:
                print(Back.WHITE + Fore.BLACK + '[TEST session %d][epoch %2d/%2d][iter %4d/%4d]'
                      % (session, current_epoch, total_epoch, step, len(test_loader)))
        accuracy = (all_preds==all_targets).sum().float() / (test_data_size * (dataset.num_classes-1)) * 100
        log.info("Epoch: {}, Accuracy: {:.3f}".format(current_epoch, accuracy))


        save_path = os.path.join(output_dir, 'cam_{}_{}.pth'.format(session, current_epoch))
        checkpoint = {'epoch': current_epoch + 1,
                      'model': cam_model.module().state_dict() if mGPU else cam_model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, save_path)
        print(Back.WHITE + Fore.BLACK + 'Model saved: %s' % (save_path))

    print(Back.GREEN + Fore.BLACK + 'Train finished.')
