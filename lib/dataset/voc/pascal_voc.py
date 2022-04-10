import os
import re
import pickle
import uuid
import cv2 as cv
import scipy.io
import json
import torch
import numpy as np
import xml.etree.ElementTree as ET
import dataset.utils as utils
from dataset.image_dataset import ImageDataset
from dataset.voc.voc_eval import voc_eval
from dataset.voc.dis_eval import dis_eval
from config import cfg

_CLASSES = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

class PascalVoc(ImageDataset):
    def __init__(self, image_set, year, params, only_classes=False):
        ImageDataset.__init__(self, 'voc_' + year + '_' + image_set, params)
        self._image_set = image_set
        self._year = year
        self._devkit_path = params['devkit_path']
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        assert os.path.exists(self._data_path), \
            'Path to data does not exist: {}'.format(self._data_path)
        self._classes = _CLASSES
        if not only_classes:
            self._class_index = dict(zip(self.classes, range(self.num_classes)))
            self._image_index = self._load_image_index()
            #self._image_index = self._image_index[:1002]
            self._image_data = self._load_image_data()
            self._salt = str(uuid.uuid4())
            self._comp_id = 'comp1'

            # PASCAL specific config options
            self.config = {'cleanup': True,
                        'use_salt': False,
                        'use_diff': False,
                        'matlab_eval': False}

    def _load_object_proposals(self, image_data):
        if cfg.TRAIN.PROPOSAL_TYPE == 'SS':
            print("Loading selective search boxes")
            ss_filename = 'voc_'+self._year+'_'+self._image_set+'.mat'
            ss_file_path = os.path.join(self._devkit_path, 'selective_search_data', ss_filename)
            assert os.path.exists(ss_file_path), 'selective search data file does not exist: {}'.format(ss_file_path)
            ss_voc2007_trainval = scipy.io.loadmat(ss_file_path)
            ss_boxes = ss_voc2007_trainval['boxes'][0]
            ss_boxes = [item[:3000] for item in ss_boxes] #select upto 2000 proposals from an image
            for i, data in enumerate(image_data):
                data['ss_boxes'] = ss_boxes[i][:, (1, 0, 3, 2)] - 1
        elif cfg.TRAIN.PROPOSAL_TYPE=='EB':
            print("Loading Edge box proposals")
            eb_filename = 'voc_'+self._year+'_'+self._image_set+'.mat'
            eb_file_path = os.path.join(self._devkit_path, 'edge_box_data', eb_filename)
            assert os.path.exists(eb_file_path), 'Edge box data file does not exist: {}'.format(eb_file_path)
            eb_voc2007_trainval = scipy.io.loadmat(eb_file_path)
            eb_boxes = eb_voc2007_trainval['boxes'][0]
            eb_scores = eb_voc2007_trainval['boxScores'][0]
            eb_boxes = [item[:3000] for item in eb_boxes]
            for i, data in enumerate(image_data):
                data['ss_boxes'] = eb_boxes[i][:, (1, 0, 3, 2)] - 1
                data['box_scores'] = eb_scores[i][:3000]
        elif cfg.TRAIN.PROPOSAL_TYPE=='CAM_SS':
            print("Loading CAM overlapping SS proposals")
            cam_ss_filename = 'gradCAM_440' + self._image_set
            file_path = os.path.join(self._devkit_path, 'cam_data', cam_ss_filename + '.pt')
            assert os.path.exists(file_path), 'Curated CAM box data file does not exist: {}'.format(file_path)
            cam_proposals = torch.load(file_path)
            for i, data in enumerate(image_data):
                data['ss_boxes'] = cam_proposals[self._image_index[i]][:, :4].cpu().numpy()
                data['box_labels'] = cam_proposals[self._image_index[i]][:, 4].cpu().numpy()
        else:
            raise ValueError('Proposal type "{}" is not defined!'.format(cfg.TRAIN.PROPOSAL_TYPE))
        return image_data


    def image_path_at(self, id):
        image_path = os.path.join(self._data_path, 'JPEGImages', str(id) + '.jpg')
        assert os.path.exists(image_path), \
            'Image Path does not exist: {}'.format(image_path)
        return image_path


    def _load_image_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = []
            for id in f.readlines():
                _tmp = re.sub(r'\s+', ' ', id).strip().split(' ')
                if len(_tmp) == 1:
                    image_index.append(_tmp[0])
                elif len(_tmp) > 1:
                    if _tmp[1] == '0' or _tmp[1] == '1': image_index.append(_tmp[0])
                else:
                    raise ValueError('Unknown string format: %s' % (id))

        return image_index
        
    def _load_annotation(self, idx, id):
        img_path = self.image_path_at(id)
        img_size = cv.imread(img_path).shape
        file_name = os.path.join(self._data_path, 'Annotations', id + '.xml')
        tree = ET.parse(file_name)
        objects = tree.findall('object')
        objects_count = len(objects)
        
        boxes = np.zeros((objects_count, 4), dtype=np.uint16)
        is_difficult = np.zeros((objects_count), dtype=np.int32)
        is_truncated = np.zeros((objects_count), dtype=np.int32)
        gt_classes = np.zeros((objects_count), dtype=np.int32)
        overlaps = np.zeros((objects_count, self.num_classes), dtype=np.float32)
        areas = np.zeros((objects_count), dtype=np.float32)
        
        for index, obj in enumerate(objects):
            bndbox = obj.find('bndbox')
            # Start coord is 0
            x1 = int(bndbox.find('xmin').text) - 1
            y1 = int(bndbox.find('ymin').text) - 1
            x2 = int(bndbox.find('xmax').text) - 1
            y2 = int(bndbox.find('ymax').text) - 1
            boxes[index, :] = [x1, y1, x2, y2]
            
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            is_difficult[index] = difficult
            
            truncated = obj.find('truncated')
            truncated = 0 if truncated is None else int(truncated.text)
            is_truncated[index] = truncated
            
            cls = self._class_index[obj.find('name').text.lower().strip()]
            gt_classes[index] = cls
            overlaps[index, cls] = 1.0
            areas[index] = (x2 - x1 + 1) * (y2 - y1 + 1)
            
        utils.validate_boxes(boxes, width=img_size[1], height=img_size[0])
        return {'index': idx,
                'id': str(id),
                'path': img_path,
                'width': img_size[1],
                'height': img_size[0],
                'boxes': boxes, 
                'gt_is_difficult': is_difficult, 
                'gt_is_truncated': is_truncated, 
                'gt_classes': gt_classes, 
                'gt_overlaps': overlaps, 
                'gt_areas': areas,
                'flipped': False}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output', log=None):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for _, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, self._image_set, ovthresh=0.5,
                use_07_metric=use_07_metric, log=log)
            aps += [ap]
            log.info('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        log.info('Mean AP = {:.4f}'.format(np.mean(aps)))
        log.info('~~~~~~~~')
        #log.info('Results:')
        #for ap in aps:
        #    log.info('{:.3f}'.format(ap))
        #log.info('{:.3f}'.format(np.mean(aps)))
        log.info('~~~~~~~~')
        log.info('')
        log.info('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _eval_discovery(self, output_dir, log):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        corlocs = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            corloc = dis_eval(
                filename, annopath, imagesetfile, cls, cachedir, self._image_set, ovthresh=0.5, log=log)
            corlocs += [corloc]
            log.info('CorLoc for {} = {:.4f}'.format(cls, corloc))
            with open(os.path.join(output_dir, cls + '_corloc.pkl'), 'wb') as f:
                pickle.dump({'corloc': corloc}, f)
        log.info('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
        log.info('~~~~~~~~')

    def _voc_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                    continue
            scores = np.around(dets[:, -1], 2)
            xs = np.around(dets[:, 0] + 1, 2)
            ys = np.around(dets[:, 1] + 1, 2)
            ws = np.around(dets[:, 2] - xs + 1, 2)
            hs = np.around(dets[:, 3] - ys + 1, 2)
            results.extend(
                [{'image_id': int(index),
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def write_all_results_file(self, all_boxes):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        result_filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(result_filedir):
            os.makedirs(result_filedir)
        res_file = 'results.json'
        res_file = os.path.join(result_filedir, res_file)
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes - 1))
            voc_cat_id = self._class_index[cls]
            results.extend(self._voc_results_one_category(all_boxes[cls_ind], voc_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir, log):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir, log)
        if self.config['matlab_eval']:
            #self._do_matlab_eval(output_dir)
            raise NotImplementedError
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def evaluate_discovery(self, all_boxes, output_dir, log):
        self._write_voc_results_file(all_boxes)
        self._eval_discovery(output_dir, log)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True