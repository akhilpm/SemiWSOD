import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg
from lib.model.rpn.rpn import _RPN
#from model.roi.roi_pool import ROIPool
#from model.roi.roi_align import ROIAlign
from torch.distributions.categorical import Categorical
from torchvision.ops import RoIPool
from torchvision.ops import RoIAlign
from lib.model.rpn.proposal_target_layer import _ProposalTargetLayer
from utils.net_utils import smooth_l1_loss
from utils.bbox_transform import bbox_transform_inv, clip_boxes, bbox_overlaps_batch
from torchvision.ops import nms
#from kmeans_pytorch import kmeans
from scipy.special import softmax

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, class_agnostic, out_depth):
        super().__init__()
        self.n_classes = num_classes
        self.class_agnostic = class_agnostic
        self.regression_weights = (10., 10., 5., 5.)

        self.RCNN_rpn = _RPN(out_depth)

        self.RCNN_proposal_target = _ProposalTargetLayer(self.regression_weights)

        if cfg.GENERAL.POOLING_MODE == 'pool':
            self.RCNN_roi_layer = RoIPool(cfg.GENERAL.POOLING_SIZE, 1.0/16.0)
        elif cfg.GENERAL.POOLING_MODE == 'align':
            self.RCNN_roi_layer = RoIAlign(cfg.GENERAL.POOLING_SIZE, 1.0/16.0, 0)
        else:
            raise ValueError('There is no implementation for "{}" ROI layer'
                             .format(cfg.GENERAL.POOLING_MODE))

    def forward(self, im_data, im_info, gt_boxes, im_labels, epoch, flag=None, real_gt_boxes=None):
        if self.training:
            assert gt_boxes is not None

        batch_size = im_data.size(0)
        debug_info = {}

        base_feature = self.RCNN_base(im_data)
        if self.training and cfg.TRAIN.USE_SS_GTBOXES:
            sampled_gt_boxes, num_sampled_boxes = self.sample_with_overlap_softmax(gt_boxes, im_labels, im_info, epoch, flag, real_gt_boxes)
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feature, im_info, sampled_gt_boxes)
            debug_info['sampled_gt_boxes'] = sampled_gt_boxes
            debug_info['num_sampled_boxes'] = num_sampled_boxes
            #debug_info['summary_record'] = summary_record
        else:
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feature, im_info, gt_boxes)
        #debug_info['rpn_boxes'] = rois # rois \in B * 2000 * 5 [batch_id, xmin, ymin, xmax, ymax]

        if self.training:
            if cfg.TRAIN.USE_SS_GTBOXES:
                roi_data = self.RCNN_proposal_target(rois, sampled_gt_boxes)
            else:
                roi_data = self.RCNN_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_target = roi_data # B * 256 * 5(batch_id, [box]), B * 256, B * 256  * 4
            debug_info['rcnn_boxes'] = rois

            rois_label = rois_label.view(-1).long()
            rois_target = rois_target.view(-1, rois_target.size(2))

        pooled_feat = self.RCNN_roi_layer(base_feature, rois.view(-1, 5)) #shape: B.256 * 512 * 7 * 7
        #mask = torch.ones(pooled_feat.size(0), 1, cfg.GENERAL.POOLING_SIZE, cfg.GENERAL.POOLING_SIZE).to(gt_boxes.device) * 0.5
        #mask = torch.bernoulli(mask.data)
        #mask = mask.expand(pooled_feat.size())
        #pooled_feat = pooled_feat * mask


        # feed pooled features to top model
        pooled_feat = self._feed_pooled_feature_to_top(pooled_feat) # B.256 * 4096 for vgg

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat) # B.256 * 4.#classes / B.2000 * 4.#classes(in test mode)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4) # B.256 * #classes * 4
            gather_idx = rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, gather_idx)
            bbox_pred = bbox_pred_select.squeeze(1) # B.256 * 4

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat) # shape: B.256 * (C+1)


        debug_info['image_labels'] = im_labels
        debug_info['image_info'] = im_info
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            pos_idx = torch.nonzero(rois_label > 0).view(-1)
            RCNN_loss_bbox = smooth_l1_loss(bbox_pred[pos_idx],
                                            rois_target[pos_idx],
                                            size_average=False)
            RCNN_loss_bbox = RCNN_loss_bbox / rois_label.numel()
            if cfg.TRAIN.USE_SS_GTBOXES:
                #gt_boxes = self.propagate_scores_all_boxes(cls_score.data, rois.data, gt_boxes, im_labels, rois_label, iou_nis)
                gt_boxes = self.propagate_scores_single_box(cls_score.data, rois.data, gt_boxes, im_labels, rois_label)
        else:
            cls_score = F.softmax(cls_score, 1)
            cls_score = cls_score.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_transform_inv(rois[:, :, 1:5], bbox_pred, self.regression_weights)
            bbox_pred = clip_boxes(bbox_pred, im_info, batch_size)

        del pooled_feat
        return cls_score, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, gt_boxes, debug_info


    def sample_with_overlap_softmax(self, gt_boxes, im_labels, im_info, epoch, flags, real_gt_boxes=None):
        batch_size = len(im_info)
        sampled_gt_boxes = gt_boxes.new_full((batch_size, 200, 5), 0, requires_grad=False)
        num_sampled_boxes = []
        temp = cfg.TRAIN.SOFTMAX_TEMP
        for i in range(batch_size):
            num_bboxes = int(im_info[i, 3].item())
            start_index = 0
            if flags[i]:
                num_real_gt_boxes = int(im_info[i, 4].item())
                sampled_gt_boxes[i, start_index:start_index+num_real_gt_boxes] = real_gt_boxes[i, :num_real_gt_boxes]
                start_index += num_real_gt_boxes
                num_sampled_boxes.append(start_index)
                continue
            for j, cls in enumerate(im_labels[i]):
                boxes = gt_boxes[i, :num_bboxes]
                cls_logits = gt_boxes[i, :num_bboxes, 4 + cls].clone().detach()
                exploit = (cls_logits*temp).cpu().numpy().astype('float64')
                probs = softmax(exploit)
                index_select = np.random.multinomial(cfg.TRAIN.NUM_BOXES_PERCLASS, probs, 1)[0]
                index_select = (index_select > 0)

                end_index = start_index + index_select.sum()
                sampled_gt_boxes[i, start_index:end_index, :4] = boxes[index_select, :4]
                sampled_gt_boxes[i, start_index:end_index, 4] = cls
                start_index = end_index

            num_sampled_boxes.append(start_index)
        return sampled_gt_boxes, num_sampled_boxes


    def propagate_scores_single_box(self, cls_score, rois, gt_boxes, im_labels, rois_label):
        batch_size = gt_boxes.size(0)
        cls_score = cls_score.view(batch_size, rois.size(1), -1)
        overlaps = bbox_overlaps_batch(gt_boxes[:, :, :4], rois[:, :, 1:5])

        for i in range(batch_size):
            overlap_scores, overlap_indices = torch.max(overlaps[i], dim=1)
            overlap_scores[overlap_scores < 0.3] = 0.0
            for j, cls in enumerate(im_labels[i]):
                old_scores = gt_boxes[i, :, 4+cls]
                current_scores = cls_score[i, overlap_indices, cls]
                new_scores = (1 - overlap_scores) * old_scores + overlap_scores * current_scores
                gt_boxes[i, :, 4 + cls] = (1-cfg.TRAIN.ALPHA)*old_scores + cfg.TRAIN.ALPHA*new_scores
        return gt_boxes


    def propagate_scores_all_boxes(self, cls_score, rois, gt_boxes, im_labels, rois_label):
        batch_size = gt_boxes.size(0)
        num_rois = rois.size(1)
        cls_score = cls_score.view(batch_size, rois.size(1), -1)
        rois_label = rois_label.view(batch_size, -1)
        overlaps = bbox_overlaps_batch(rois, gt_boxes)
        overlaps[overlaps < 0.3] = 0.0
        for i in range(batch_size):
            for j, cls in enumerate(im_labels[i]):
                order = torch.argsort(cls_score[i, :, cls], 0)
                for k in range(num_rois):
                    current_scores = cls_score[i, order[k], cls]
                    overlap_scores = overlaps[i, order[k]]
                    old_scores = gt_boxes[i, :, 4 + cls]
                    new_scores = (1 - overlap_scores) * old_scores + overlap_scores * current_scores
                    gt_boxes[i, :, 4 + cls] = (1-cfg.TRAIN.ALPHA)*old_scores + cfg.TRAIN.ALPHA*new_scores
        return gt_boxes


    def _repeat_as_scalar_tensor_by_count(self, scalar_tensor, repeat_counts):
        new_tenor = []
        for i, count in enumerate(repeat_counts):
            new_tenor.append(scalar_tensor[i].repeat(count))
        return torch.cat(new_tenor).to(scalar_tensor.device)


    def _prepare_pooled_feature(self, pooled_feature):
        raise NotImplementedError
        
    def _init_modules(self):
        raise NotImplementedError
        
    def _init_weights(self):
        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01)
        normal_init(self.RCNN_cls_score, 0, 0.01)
        normal_init(self.RCNN_bbox_pred, 0, 0.001)

    def init(self):
        self._init_modules()
        self._init_weights()

