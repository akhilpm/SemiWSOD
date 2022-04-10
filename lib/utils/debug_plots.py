import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from config import cfg
from utils.bbox_transform import bbox_overlaps_batch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from torchvision.ops import nms
import matplotlib.patches as patches

def plot_debug_info(image, debug_info, idx, epoch, image_id, classes, is_training=True):
    image_copy = image.copy()
    if is_training:
        #plot Detection and GT boxes
        det_boxes = debug_info['rcnn_boxes'][idx].cpu().numpy()
        gt_boxes = debug_info['gt_boxes'].cpu().numpy()
        gt_boxes[:, :4] += cfg.VIS_OFFSET
        det_boxes[:, 1:5] += cfg.VIS_OFFSET
        im_labels = debug_info['image_labels'][idx].cpu().numpy()
        save_dir = os.path.join(debug_info['save_root_dir'], 'rcnn_boxes')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #num_boxes = debug_info['num_boxes_this_image']
        #pseudo_gt_scores = debug_info['gt_scores']
        """
        for i in range(min(20, num_boxes)):
            bbox = tuple(int(np.round(x)) for x in gt_boxes[i, :4])
            gt_scores = np.around(pseudo_gt_scores[i, labels], 2)
            thickness_scale = int(np.round(np.max(gt_scores).clip(1, 10)))
            cv2.rectangle(image, bbox[0:2], bbox[2:4], (0, 255, 255), thickness_scale)

            text_width, text_height = cv2.getTextSize(str(gt_scores), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
            box_coords = ((bbox[0], bbox[1]), (bbox[0] + text_width + 2, bbox[1] - text_height - 2))
            cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
            cv2.putText(image, str(gt_scores), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
            """

        cat_names = [classes[class_idx][:3] for class_idx in im_labels]
        cv2.putText(image, str(cat_names), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

        #det_boxes = debug_info['rcnn_boxes'][idx].cpu().numpy()
        sampled_boxes = debug_info['sampled_gt_boxes'][idx].cpu().numpy()
        sampled_boxes[:, :4] += cfg.VIS_OFFSET
        for label in im_labels:
            plot_image = image.copy()
            save_path = os.path.join(save_dir, image_id + '_' + classes[label] + '_epoch_' + str(epoch) + '_det.jpg')
            for i in range(len(sampled_boxes)):
                if sampled_boxes[i, 4]==label:
                    bbox = sampled_boxes[i, :4]
                    bbox[0:2] += 2.5 # shift the box little bit inside to improve the view when boxes overlap perfectly
                    bbox[2:4] -= 2.5
                    bbox = tuple(int(np.round(x)) for x in bbox)
                    cv2.rectangle(plot_image, bbox[0:2], bbox[2:4], (255, 255, 0), 2)
            cv2.imwrite(save_path, plot_image)
    else:

        #plot gt boxes probability as heatmap
        image = image_copy.copy()
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        labels = debug_info['image_labels'][idx].cpu().numpy()
        real_gt_boxes = debug_info['real_gt_boxes']

        save_dir = os.path.join(debug_info['save_root_dir'], 'gt_heatmap')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        height, width = image_copy.shape[0], image_copy.shape[1]
        image_info = debug_info['image_info'][idx].cpu().numpy()
        num_boxes = int(image_info[3])
        num_real_gt_boxes = int(image_info[4])
        gt_boxes = debug_info['gt_boxes']
        #iou_nis = debug_info['iou_nis'][:num_boxes].cpu().numpy()
        #iou_nis = cfg.TRAIN.C * np.sqrt(epoch/iou_nis)
        box_probs = gt_boxes[:num_boxes, 4:].cpu().numpy()
        #box_probs = box_probs + iou_nis.reshape(-1, 1)
        gt_boxes = gt_boxes.cpu().numpy()
        for i, label in enumerate(labels):
            #sampled_boxes = debug_info['sampled_gt_boxes'].cpu().numpy()
            if width>height:
                fig, ax = plt.subplots(2, 1)
            else:
                fig, ax = plt.subplots(1, 2)
            ax[0].axis('off')
            ax[1].axis('off')
            ax[0].imshow(image.astype(np.uint8))
            temp = np.zeros((height, width), dtype=np.float)
            normalizer = np.zeros((height, width), dtype=np.float) + 1e-5
            """
            for j in range(len(sampled_boxes)):
                if sampled_boxes[j, 4]==label:
                    bbox = sampled_boxes[j, :4]
                    bbox = tuple(int(np.round(x)) for x in bbox)
                    #cv2.rectangle(temp, bbox[0:2], bbox[2:4], (255, 0, 0), 5)
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='g', facecolor='none')
                    ax[1].add_patch(rect)
            """
            for j in range(num_real_gt_boxes):
                if real_gt_boxes[j, 4]==label:
                    bbox = real_gt_boxes[j, :4]
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='g', facecolor='none')
                    ax[1].add_patch(rect)
            for j in range(num_boxes):
                bbox = tuple(int(np.round(x)) for x in gt_boxes[j, :4])
                temp[bbox[1]:bbox[3]-1, bbox[0]:bbox[2]-1] += box_probs[j, label]
                #temp[bbox[1]:bbox[3] - 1, bbox[0]:bbox[2] - 1] += iou_nis[j]
                normalizer[bbox[1]:bbox[3]-1, bbox[0]:bbox[2]-1] += 1.0
            temp = temp / normalizer
            temp = temp[:-5, :-5]
            temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp)+1e-5)
            print(np.min(temp), np.max(temp))
            ax[1].imshow(temp, cmap='hot', interpolation='nearest')
            class_name = classes[label]
            save_path = os.path.join(save_dir, image_id + '_' + class_name + '_epoch_' + str(epoch) + '_map.jpg')
            fig.savefig(save_path)
            fig.clf()
            plt.close('all')

        """
        save_dir = os.path.join(debug_info['save_root_dir'], 'rpn_boxes')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        gt_boxes = debug_info['gt_boxes']


        for i, label in enumerate(labels):
            cls_real_gt_boxes = real_gt_boxes[real_gt_boxes[:, 4]==label]
            overlaps = bbox_overlaps_batch(gt_boxes[:, 0:4], cls_real_gt_boxes).squeeze(0)
            ious = overlaps.sum(dim=1).cpu().numpy()
            #ious = overlaps.max(dim=1)[0].cpu().numpy()
            scores = gt_boxes[:num_boxes, 4+label].cpu().numpy()
            #scores += iou_nis[:num_boxes]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_ylim((0.0, 1.5))
            #ax.scatter(ious, iou_nis, scores, c='r', marker='o')
            for j in range(num_boxes):
                #ax.scatter(scores[j], iou_nis[j], ious[j], c='r', marker='o')
                #ax.text(scores[j], iou_nis[j], ious[j], "%s"%(str(np.around(ious[j], 2))+','+str(np.around(scores[j], 1))+','+str(np.around(iou_nis[j], 2))), size=8, zdir='x')
                ax.scatter(scores[j], ious[j], c='r', marker='o')
                #ax.text(scores[j], ious[j], "%s" % (str(np.around(ious[j], 2)) + ',' + str(np.around(scores[j], 1))+','+str(np.around(iou_nis[j], 2))), size=8)
            ax.set_xlabel('scores')
            ax.set_ylabel('ious')
            ax.legend(["real GT IOU, score, explore"])

            save_path = os.path.join(save_dir, image_id + '_' + classes[label] + '_epoch_' + str(epoch) + '_map.jpg')
            fig.savefig(save_path)
            fig.clf()
            plt.close('all')
        """


