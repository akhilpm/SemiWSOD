import os
import sys
import time
import cv2 as cv
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from lib.model.vgg16 import VGG16
from lib.model.resnet import Resnet
from torchvision.ops import nms
#from torchvision.ops import box_convert
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
from utils.bbox_transform import bbox_overlaps_batch

watch_list = ['000012', '000017', '000019', '000021', '000026', '000036', '000089', '000102', '000121', '000130', '000198']

def find_iou(bb, BBGT):
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    area_bb = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.)
    area_BBGT = (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.)
    uni = ( area_bb + area_BBGT - inters)
    overlaps = inters / uni
    ratio = area_bb/area_BBGT
    return np.max(ratio), np.max(overlaps)


def plot_detecton_boxes(image, cls_scores, classes, dets, im_labels, real_gt_boxes=None, sampled_boxes=None):
    obj_classes = []
    for label in im_labels:
        obj_classes.append(classes[label][:3])
    text_width, text_height = cv.getTextSize(' '.join(obj_classes), cv.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
    box_coords = ((1, 1 + 10), (1 + text_width + 2, 1 + 10 - text_height - 2))
    cv.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv.FILLED)
    cv.putText(image, ' '.join(obj_classes), (1, 1 + 10), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    if real_gt_boxes is not None:
        for i in range(real_gt_boxes.shape[0]):
            bbox = tuple(int(np.round(x)) for x in real_gt_boxes[i, :4])
            cv.rectangle(image, bbox[0:2], bbox[2:4], (255, 255, 255), 6)

    if sampled_boxes is not None:
        for j in range(len(sampled_boxes)):
            bbox = tuple(int(np.round(x)) for x in sampled_boxes[j, :4])
            cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 0, 0), 3)

    sel_indices = np.argsort(-cls_scores)[:20][::-1]
    cls_scores = cls_scores[sel_indices]
    dets = dets[sel_indices]
    cNorm = colors.Normalize(vmin=min(cls_scores), vmax=max(cls_scores))
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    for i in range(dets.shape[0]):
        #bbox = tuple(int(np.round(x)) for x in dets[i, 1:5])
        scores = np.around(cls_scores[i], 2)
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        cmap = scalarMap.to_rgba(cls_scores[i])
        cmap = tuple((np.array(cmap[0:-1]) * 255).astype(np.int32))[::-1]
        cv.rectangle(image, bbox[0:2], bbox[2:4], (int(cmap[0]), int(cmap[1]), int(cmap[2])), 2)
        #cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 255, 0), 2)
        text_width, text_height = cv.getTextSize(str(scores), cv.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        box_coords = ((bbox[0], bbox[1] + 15), (bbox[0] + text_width + 2, bbox[1] + 15 - text_height - 2))
        cv.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv.FILLED)
        cv.putText(image, str(scores), (bbox[0], bbox[1] + 15), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
    return image

def clip_boxes(rois, height, width):
    rois[:, 1].clamp_(0, width-1)
    rois[:, 2].clamp_(0, height-1)
    rois[:, 3].clamp_(0, width-1)
    rois[:, 4].clamp_(0, height-1)
    return rois

def hyp_test(dataset, net, class_agnostic, load_dir, session, epoch, log, add_params):
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
    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_test)

    if 'data_path' in add_params: cfg.DATA_DIR = add_params['data_path']
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

    #model_path = os.path.join(cfg.DATA_DIR, load_dir, net, ds_name, 'frcnn_{}_{}.pth'.format(session, epoch))
    #log.info(Back.WHITE + Fore.BLACK + 'Loading model from %s' % (model_path))
    #checkpoint = torch.load(model_path, map_location=device)
    #faster_rcnn.load_state_dict(checkpoint['model'])
    #log.info('Done.')

    start = time.time()

    faster_rcnn.eval()
    debug_dir = os.path.join(cfg.DATA_DIR, 'debug', 'session_' + str(session))

    # load the scores of this epoch
    if dataset._image_set == 'trainval':
        save_dir = os.path.join(debug_dir, 'ss_box_scores')
        save_path = os.path.join(save_dir, 'score_epoch_' + str(epoch) + '.pt')
        pseudo_GT_scores = torch.load(save_path)
        save_dir = os.path.join(debug_dir, 'sampled_boxes')
        save_path = os.path.join(save_dir, 'sampled_boxes_' + str(epoch) + '.pt')
        sampled_boxes = torch.load(save_path)
        #save_dir = os.path.join(debug_dir, 'iou_nis')
        #save_path = os.path.join(save_dir, 'iou_nis_' + str(epoch) + '.pt')
        #iou_nis = torch.load(save_path)

    save_det_dir = os.path.join(debug_dir, 'plot_boxes')
    if not os.path.exists(save_det_dir):
        os.makedirs(save_det_dir)

    all_ious = []
    all_criterias = []
    for i, data in enumerate(loader):
        #image_data = data[0].to(device)
        image_info = data[1]#.to(device)
        gt_boxes = data[2].to(device)
        image_labels = data[3]
        image_ids = data[4]
        real_gt_boxes = data[5].to(device)
        num_gt_boxes = int(image_info[0, 3])

        #base_feature = faster_rcnn.RCNN_base(image_data)
        #rois = torch.zeros(num_gt_boxes, 5).to(device)
        #rois[:, 1:5] = gt_boxes[0, :num_gt_boxes, :4]
        #pooled_feat = faster_rcnn.RCNN_roi_layer(base_feature, rois)
        #pooled_feat = faster_rcnn._feed_pooled_feature_to_top(pooled_feat)
        #cls_score = faster_rcnn.RCNN_cls_score(pooled_feat).detach()


        if len(real_gt_boxes)>0:
            sampled_boxes_this_image = sampled_boxes[image_ids[0]]
            for j, cls in enumerate(image_labels[0]):
                #if cls==5:
                actual_gt_boxes = real_gt_boxes[0][real_gt_boxes[0, :, 4] == cls, :4]
                sampled_boxes_this_class = sampled_boxes_this_image[sampled_boxes_this_image[:, 4]==cls, :4]
                max_box = torch.argmax(pseudo_GT_scores[image_ids[0]][:num_gt_boxes, cls])
                iou, corloc = find_iou(gt_boxes[0, max_box, :4].cpu().numpy(), actual_gt_boxes.cpu().numpy())
                all_ious.append([iou, corloc])

                overlaps = bbox_overlaps_batch(actual_gt_boxes, sampled_boxes_this_class).squeeze(0)
                #overlaps = bbox_overlaps_batch(actual_gt_boxes, gt_boxes[0, :num_gt_boxes, :4]).squeeze(0)
                criteria = (overlaps >= 0.5).sum(dim=1).cpu().numpy()
                criteria = (criteria >=1)
                all_criterias.append(criteria)


        if i % 200 == 0:
            print("Plotting for {} th image".format(i))
            watch_list.append(image_ids[0])

            gt_boxes /= image_info[0][2].item()
            gt_boxes = gt_boxes[0, :num_gt_boxes].cpu().numpy()
            cls_score = pseudo_GT_scores[image_ids[0]][:num_gt_boxes]
            num_real_gt_boxes = int(image_info[0, 4])
            real_gt_boxes[0, :, :4] /= image_info[0][2].item()
            real_gt_boxes = real_gt_boxes[0, :num_real_gt_boxes]
            sampled_boxes_this_image = sampled_boxes[image_ids[0]]

            for j, cls in enumerate(image_labels[0]):
                image = cv.imread(dataset.image_path_at(data[4][0]))
                sampled_boxes_this_class = sampled_boxes_this_image[sampled_boxes_this_image[:, 4]==cls, :4].cpu().numpy()
                sampled_boxes_this_class /= image_info[0][2].item()
                real_gt_boxes_this_class = real_gt_boxes[real_gt_boxes[:, 4] ==cls, :4].cpu().numpy()
                image = plot_detecton_boxes(image, cls_score[:, cls].cpu().numpy(), dataset.classes, gt_boxes, image_labels[0].cpu().numpy(), real_gt_boxes_this_class)
                class_name = dataset.classes[cls]
                save_det_path = os.path.join(save_det_dir, image_ids[0] + '_' + class_name + '_epoch_' + str(epoch) + '_det.jpg')
                cv.imwrite(save_det_path, image)



    all_ious = np.array(all_ious)
    no_overlap = np.sum(all_ious[:, 1]==0)
    greater = np.sum(all_ious[:, 0]>1.0)
    smaller = len(all_ious) - no_overlap - greater
    corloc = np.sum(all_ious[:, 1]>=0.5)
    log.info("CorLoc: {:.3f} {}/{}".format(float(corloc)/len(all_ious), corloc, len(all_ious)))
    log.info("No overlap: {:.3f} {}/{}".format(float(no_overlap)/len(all_ious), no_overlap, len(all_ious)))
    log.info("No of small proposals: {:.3f} {}/{}".format(float(smaller) / len(all_ious), smaller, len(all_ious)))
    log.info("No of large proposals: {:.3f} {}/{}".format(float(greater) / len(all_ious), greater, len(all_ious)))

    all_criterias = np.hstack(all_criterias)
    log.info("Covered: {} {}/{}".format(float(all_criterias.sum())/len(all_criterias), all_criterias.sum(), len(all_criterias)))

    end = time.time()
    log.info(Back.GREEN + Fore.BLACK + 'Plot time: %.4fs.' % (end - start))
