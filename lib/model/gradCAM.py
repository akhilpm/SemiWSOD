import torch
import torch.nn.functional as F
import numpy as np


def find_vgg_layer(arch, target_layer_name='features_29'):
    hierarchy = target_layer_name.split('_')
    if len(hierarchy)>=1:
        target_layer = arch.cls_net.features
    if len(hierarchy)==2:
        target_layer = target_layer[int(hierarchy[1])]
    else:
        raise ValueError("target layer not defined")
    return target_layer

class gradCAM(object):
    def __init__(self, model, verbose=False):
        self.gradients = dict()
        self.activations = dict()
        self.model_arch = model

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0].detach()
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output.detach()
            return None

        target_layer = find_vgg_layer(self.model_arch)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, image_label=None, retain_graph=False):
        b, c, h, w = x.size()
        logits, target = self.model_arch(x, torch.tensor([image_label]))
        if image_label is None:
            score = logits[:, logits.max(1)[-1]].squeeze()
        else:
            score = logits[:, (image_label-1).long()].squeeze(0)
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        weights = gradients.view(b, k, -1).mean(2)
        weights = weights.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map = saliency_map.squeeze().squeeze().cpu().numpy()
        saliency_map = np.uint8(255 * saliency_map)
        return saliency_map, logits
