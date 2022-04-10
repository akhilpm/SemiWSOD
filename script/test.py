import os
import sys
import time
import cv2 as cv
import pickle
import torch
import shutil
import numpy as np
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from lib.model.vgg16 import VGG16
from lib.model.resnet import Resnet
from utils.debug_plots import plot_debug_info
#from _C import nms
from torchvision.ops import nms
import matplotlib.pyplot as plt
from utils.bbox_transform import bbox_overlaps_batch

watch_list = ['000012', '000017', '000019', '000021', '000026', '000036', '000089', '000102', '000121', '000130', '000198']

def inverse_transform(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x = image.new(*image.size())
    x[0, :, :] = image[0, :, :] * std[0] + mean[0]
    x[1, :, :] = image[1, :, :] * std[1] + mean[1]
    x[2, :, :] = image[2, :, :] * std[2] + mean[2]
    x = (x * 255).clamp_(0, 255)
    x = x.permute(1, 2, 0).cpu().numpy()
    return x

def plot_detecton_boxes(image, debug_info, dets, class_name):
    image_info = debug_info['image_info'][0].cpu().numpy()
    real_gt_boxes = debug_info['real_gt_boxes']
    num_real_gt_boxes = int(image_info[4])
    #real_gt_boxes /= debug_info['image_info'][0][2].item()

    overlaps = bbox_overlaps_batch(dets[:, :4], real_gt_boxes[:, :4]).squeeze(0)
    box_label = (overlaps >= 0.5).sum(dim=1)
    dets = dets.cpu().numpy()
    for i in range(np.minimum(100, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score<0.3:
            continue
        if box_label[i] == 0:
            cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 0, 255), 2)
        else:
            cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 255, 0), 2)
        text_width, text_height = \
        cv.getTextSize('{:s}: {:.3f}'.format(class_name, score), cv.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        box_coords = ((bbox[0], bbox[1] + 15), (bbox[0] + text_width + 2, bbox[1] + 15 - text_height - 2))
        cv.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv.FILLED)
        cv.putText(image, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
    for i in range(num_real_gt_boxes):
        bbox = tuple(int(np.round(x)) for x in real_gt_boxes[i, :4].cpu())
        cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 255, 255), 3)
    return image


def get_multi_scale_images(image_id, params, year="2007"):
    data_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit'+year, 'VOC'+year)
    image_path = os.path.join(data_path, 'JPEGImages', str(image_id) + '.jpg')
    cv_image = cv.imread(image_path)
    multi_scale_images = []
    scale_factors = []
    for scale in cfg.GENERAL.SCALES:
        original_image = cv_image.copy()
        image = original_image[:, :, ::-1] # BGR -> RGB
        image = image.astype(np.float32, copy=True) / (255.0 if params['image_range']==1 else 1.0)
        image = (image - np.array([[params['mean']]])) / np.array([[params['std']]])
        image_shape = image.shape
        im_size_max = np.max(image_shape[:2])
        im_size_min = np.min(image_shape[:2])
        im_scale = float(scale) / im_size_min
        if im_scale * im_size_max > cfg.GENERAL.MAX_IMG_SIZE:
            im_scale = float(cfg.GENERAL.MAX_IMG_SIZE) / im_size_max
        image = cv.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
        image = image.astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        multi_scale_images.append(image)
        scale_factors.append(im_scale)
    return multi_scale_images, scale_factors


def test(dataset, net, class_agnostic, load_dir, session, epoch, log, add_params):
    log.info("============== Testing EPOCH {} =============".format(epoch))
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    print(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))

    if 'cfg_file' in add_params:
        update_config_from_file(add_params['cfg_file'])

    log.info(Back.WHITE + Fore.BLACK + 'Using config:')
    log.info('GENERAL:')
    log.info(cfg.GENERAL)
    log.info('TEST:')
    log.info(cfg.TEST)
    log.info('RPN:')
    log.info(cfg.RPN)

    # TODO: add competition mode
    dataset_name = dataset.split('_')[0]
    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                        collate_fn=collate_test)

    #if 'data_path' in add_params: cfg.DATA_DIR = add_params['data_path']
    output_dir = os.path.join(cfg.DATA_DIR, 'output', net, ds_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(Back.CYAN + Fore.BLACK + 'Output directory: %s' % (output_dir))

    if net == 'vgg16':
        faster_rcnn = VGG16(dataset.num_classes, class_agnostic=class_agnostic)
    elif net.startswith('resnet'):
        num_layers = net[6:]
        faster_rcnn = Resnet(num_layers, dataset.num_classes, class_agnostic=class_agnostic)
    else:
        raise ValueError(Back.RED + 'Network "{}" is not defined!'.format(net))

    faster_rcnn.init()
    faster_rcnn.to(device)

    model_path = os.path.join(cfg.DATA_DIR, load_dir, net, ds_name, 
                              'frcnn_{}_{}.pth'.format(session, epoch))
    log.info(Back.WHITE + Fore.BLACK + 'Loading model from %s' % (model_path))
    checkpoint = torch.load(model_path, map_location=device)
    faster_rcnn.load_state_dict(checkpoint['model'])
    log.info('Done.')

    start = time.time()
    max_per_image = 100

    all_boxes = [[torch.empty(0, 5).numpy() for _ in range(len(dataset))] for _ in range(dataset.num_classes)]

    faster_rcnn.eval()
    debug_dir = os.path.join(cfg.DATA_DIR, 'debug', 'session_'+str(session))
    #if epoch==1 and os.path.exists(debug_dir):
    #    shutil.rmtree(debug_dir)

    #load the scores of this epoch
    if dataset._image_set=='trainval':
        save_dir = os.path.join(debug_dir, 'ss_box_scores')
        save_path = os.path.join(save_dir, 'score_epoch_' + str(epoch) + '.pt')
        pseudo_GT_scores = torch.load(save_path)
        save_dir = os.path.join(debug_dir, 'sampled_boxes')
        save_path = os.path.join(save_dir, 'sampled_boxes_' + str(epoch) + '.pt')
        sampled_boxes = torch.load(save_path)
        #watch_list = watch_list + dataset._image_index[0::200]
        #save_dir = os.path.join(debug_dir, 'iou_nis')
        #save_path = os.path.join(save_dir, 'iou_nis_' + str(epoch) + '.pt')
        #iou_nis = torch.load(save_path)

    save_det_dir = os.path.join(debug_dir, 'detection_boxes')
    if not os.path.exists(save_det_dir):
        os.makedirs(save_det_dir)

    watch_list = dataset._image_index[:15]
    for i, data in enumerate(loader):
        image_data = data[0].to(device)
        image_info = data[1].to(device)
        gt_boxes = data[2].to(device)
        image_labels = data[3]
        image_ids = data[4]
        real_gt_boxes = data[5].to(device)

        det_tic = time.time()
        with torch.no_grad():
            if cfg.TEST.MULTI_SCALE_TESTING:
                cls_score, bbox_pred = [], []
                multi_scale_images, scale_factor = get_multi_scale_images(image_ids[0], add_params)
                for k, image in enumerate(multi_scale_images):
                    image_data = image.unsqueeze(0).to(device)
                    cls_score_this_scale, bbox_pred_this_scale, _, _, _, _, _, debug_info = faster_rcnn(image_data, image_info, gt_boxes, image_labels, None)
                    cls_score.append(cls_score_this_scale)
                    bbox_pred_this_scale /= scale_factor[k]
                    bbox_pred.append(bbox_pred_this_scale)
                cls_score = torch.cat(cls_score, dim=1).to(device)
                bbox_pred = torch.cat(bbox_pred, dim=1).to(device)
            else:
                cls_score, bbox_pred, _, _, _, _, _, debug_info = faster_rcnn(image_data, image_info, gt_boxes, image_labels, None)
                bbox_pred /= image_info[0][2].item()
        #if i%200==0:
        #    watch_list.append(image_ids[0])

        for j, id in enumerate(image_ids):
            if id in watch_list:
                if dataset._image_set == 'trainval':
                    gt_boxes[j, :, 4:] = pseudo_GT_scores[id]
                    #debug_info['iou_nis'] = iou_nis[id]
                    debug_info['sampled_gt_boxes'] = sampled_boxes[id]
                    debug_info['gt_boxes'] = gt_boxes[j]
                    debug_info['save_root_dir'] = debug_dir
                    debug_info['cls_score'] = cls_score.squeeze()
                    debug_info['num_classes'] = dataset.num_classes
                    debug_info['bbox_pred'] = bbox_pred.squeeze()
                    image = inverse_transform(image_data[j])
                    image = cv.cvtColor(image.astype(np.uint8), cv.COLOR_RGB2BGR)
                    #image = cv.imread(dataset.image_path_at(data[4][0]))
                    debug_info['real_gt_boxes'] = real_gt_boxes[j] #/ image_info[0][2].item()
                    plot_debug_info(image, debug_info, j, epoch, id, dataset.classes, is_training=False)
        scores = cls_score.squeeze()
        bbox_pred = bbox_pred.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic

        misc_tic = time.time()
        for j in range(1, dataset.num_classes):
            inds = torch.nonzero(scores[:,j] > 0.01).view(-1)
            #if j not in image_labels[0]:
            #    inds = torch.empty(0)
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, descending=True)
                if class_agnostic:
                    cls_boxes = bbox_pred[inds, :]
                else:
                    cls_boxes = bbox_pred[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                keep = nms(cls_boxes, cls_scores, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                all_boxes[j][i] = cls_dets.cpu().numpy()
                #if image_ids[0] in watch_list:
                #    image = plot_detecton_boxes(image, debug_info, cls_dets, dataset.classes[j])
            else:
                all_boxes[j][i] = torch.empty(0, 5).numpy()
        #if image_ids[0] in watch_list:
        #    save_det_path = os.path.join(save_det_dir, image_ids[0] + '_' + '_epoch_' + str(epoch) + '_det.jpg')
        #    cv.imwrite(save_det_path, image)
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in range(1, dataset.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, dataset.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, len(dataset), detect_time, nms_time))
        sys.stdout.flush()

        #if i > 110:
        #    break
            
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    #dataset.write_all_results_file(all_boxes)

    log.info('\nEvaluating detections...')
    dataset.evaluate_detections(all_boxes, output_dir, log)

    # TODO: Add txt file with result info ?

    end = time.time()
    log.info(Back.GREEN + Fore.BLACK + 'Test time: %.4fs.' % (end - start))

    #log.info('\nEvaluating corLoc....')
    #dataset.evaluate_discovery(all_boxes, output_dir, log)
