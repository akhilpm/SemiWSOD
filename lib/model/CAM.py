import torch
import torch.nn.functional as F
import numpy as np



class CAM(object):
    def __init__(self, model, verbose=False):
        self.gradients = dict()
        self.activations = dict()
        self.model_arch = model

    def __call__(self, x, image_label=None):
        b, c, h, w = x.size()
        self.model_arch.zero_grad()
        logits, target, saliency_map = self.model_arch(x, torch.tensor([image_label]))
        if image_label is None:
            saliency_map = saliency_map[logits.max(1)[-1]].unsqueeze(0).unsqueeze(0)
        else:
            saliency_map = saliency_map[(image_label-1).long()].unsqueeze(0).unsqueeze(0)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map = saliency_map.squeeze().squeeze().cpu().numpy()
        saliency_map = np.uint8(255 * saliency_map)
        return saliency_map, logits
