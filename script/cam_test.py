import numpy as np
import os
import torch
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from lib.model.clf_net import Cls_Net
from lib.model.gradCAM import gradCAM
from lib.model.gap_cls_net import GAP_Net
from lib.model.CAM import CAM
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from utils.bbox_transform import bbox_overlaps_batch
import cv2 as cv


watch_list = ['000012', '000017', '000019', '000021', '000026', '000036', '000089', '000102', '000121', '000130', '000198']

def bbox_inside(box, gt_boxes):
    xmin = (gt_boxes[:, 0] <= box[0])
    ymin = (gt_boxes[:, 1] <= box[1])
    xmax = (gt_boxes[:, 2] >= box[2])
    ymax = (gt_boxes[:, 3] >= box[3])
    return (xmin & ymin & xmax & ymax)

def plot_heatmap(height, width, image, contours, binary_mask, save_path):
    image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
    if width > height:
        fig, ax = plt.subplots(2, 1)
    else:
        fig, ax = plt.subplots(1, 2)
    ax[0].axis('off')
    ax[1].axis('off')
    image_rgb = cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB)
    cv.drawContours(image_rgb, contours, -1, (0, 255, 0), 3)
    ax[0].imshow(image_rgb)
    ax[1].imshow(binary_mask, cmap='jet', interpolation='nearest')
    fig.savefig(save_path)
    fig.clf()
    plt.close('all')

def cam_test(dataset, net, load_dir, session, epoch, log, cam_type, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    #fetch dataset
    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_test)
    model_path = os.path.join(cfg.DATA_DIR, load_dir, net, ds_name, 'cam_{}_{}.pth'.format(session, epoch))
    log.info(Back.WHITE + Fore.BLACK + 'Loading model from %s' % (model_path))
    checkpoint = torch.load(model_path, map_location=device)
    if cam_type=='gradcam':
        cam_model = Cls_Net(dataset.num_classes-1)
    else:
        cam_model = GAP_Net(dataset.num_classes-1)
    cam_model.to(device)
    cam_model.load_state_dict(checkpoint['model'])
    if cam_type=='gradcam':
        grad_cam = gradCAM(cam_model)
    else:
        grad_cam = CAM(cam_model)

    save_root_dir = os.path.join(cfg.DATA_DIR, 'debug', 'session_' + str(session))
    save_dir = os.path.join(save_root_dir, 'gt_heatmap')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_size = len(dataset)
    cam_model.eval()
    all_criterias = []
    count_of_selected_boxes = []
    chosen_boxes = {}
    for i in range(data_size):
        chosen_boxes[dataset._image_index[i]] = []
    for step, data in enumerate(loader):
        image_data = data[0].to(device)
        image_info = data[1].to(device)
        gt_boxes = data[2].to(device)
        image_labels = data[3]
        image_ids = data[4]
        real_gt_boxes = data[5].to(device)
        num_boxes = int(image_info[0, 3])
        num_real_gt_boxes = int(image_info[0, 4])
        if step%200==0:
            print("Step #: {}".format(step))
            #watch_list.append(image_ids[0])
        #if image_ids[0] in watch_list:
        #    height, width = image_data.size(2),  image_data.size(3)
        #    image = cv.imread(dataset.image_path_at(image_ids[0]))
        #    image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
        total_selected_boxes = np.zeros(num_boxes, dtype=np.bool)
        gt_boxes = gt_boxes[0, :num_boxes, :4]

        proposals = np.empty([0, 4], dtype=np.float32)
        proposal_labels = np.zeros(num_boxes)
        for i, label in enumerate(image_labels[0]):
            saliency, logits = grad_cam(image_data, label)
            max_act = np.max(saliency)
            binary_mask = (saliency>=0.2*max_act)
            binary_mask = np.uint8(255 * binary_mask)
            contours, hierarchy = cv.findContours(cv.UMat(binary_mask), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
            class_selection = np.zeros(num_boxes, dtype=np.bool)
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                inclusion = bbox_inside(np.array([x, y, x+w, y+h]), gt_boxes.cpu().numpy())
                total_selected_boxes = total_selected_boxes | inclusion
                class_selection = class_selection | inclusion
                proposals = np.append(proposals, np.array([x, y, x+w, y+h], dtype=np.float32).reshape(1, -1), axis=0)
                #cv.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
            proposal_labels[class_selection] = label
        overlaps = bbox_overlaps_batch(gt_boxes, torch.from_numpy(proposals).to(device)).squeeze(0)
        #overlaps = bbox_overlaps_batch(gt_boxes, real_gt_boxes[0, :num_real_gt_boxes, :4]).squeeze(0)
        covered = (overlaps>0.1).sum(dim=1).cpu().numpy()
        total_selected_boxes = total_selected_boxes | (covered>=1)
        selected_boxes = gt_boxes[total_selected_boxes]
        overlaps = bbox_overlaps_batch(real_gt_boxes[0, :num_real_gt_boxes, :4], selected_boxes).squeeze(0)
        criteria = (overlaps >= 0.5).sum(dim=1).cpu().numpy()
        criteria = (criteria >= 1)
        selected_boxes = gt_boxes[total_selected_boxes]
        proposal_labels = proposal_labels[total_selected_boxes]
        if total_selected_boxes.sum()==0:
            log.info("No boxes selected for image: {}".format(image_ids[0]))
            selected_boxes = gt_boxes
            proposal_labels = torch.zeros(num_boxes).to(device)
        #image = cv.imread(dataset.image_path_at(image_ids[0]))
        #save_path = os.path.join(save_dir, image_ids[0] + '_' + dataset.classes[label] + '_epoch_' + str(epoch) + '_map.jpg')
        #plot_heatmap(image_data.size(2), image_data.size(3), image, contours, binary_mask, save_path)
        all_criterias.append(criteria)
        chosen_boxes[image_ids[0]] = torch.zeros((len(selected_boxes), 5)).to(device)
        chosen_boxes[image_ids[0]][:, 4] = torch.from_numpy(proposal_labels).to(device)
        chosen_boxes[image_ids[0]][:, :4] = selected_boxes / image_info[0][2].item()
        count_of_selected_boxes.append(len(selected_boxes))

    save_path = os.path.join(save_root_dir, 'curated_boxes.pt')
    torch.save(chosen_boxes, save_path)
    count_of_selected_boxes = np.array(count_of_selected_boxes)
    log.info("Mean count: {}, Min: {}, Max: {}".format(count_of_selected_boxes.mean(), count_of_selected_boxes.min(), count_of_selected_boxes.max()))
    log.info("No of zero box selections: {}".format(np.sum(count_of_selected_boxes==0)))
    all_criterias = np.hstack(all_criterias)
    log.info("Covered: {} {}/{}".format(float(all_criterias.sum()) / len(all_criterias), all_criterias.sum(), len(all_criterias)))