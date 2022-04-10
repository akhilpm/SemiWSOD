from torchvision.models import vgg16
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from config import cfg

class GAP_Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = 512
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([np.arange(1, num_classes+1)])
        net = vgg16(pretrained=True)
        self.features = nn.Sequential(*list(net.features)[:-1])
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, im_data, im_labels=None):
        batch_size = im_data.size(0)
        features = self.features(im_data)
        cam = None
        if not self.training:
            fmap_size = features.size(2)
            gap_activations = features.detach().clone()
            gap_activations = gap_activations.view(self.in_features, -1)
            weight = self.fc.weight.data
            cam = torch.matmul(weight, gap_activations)
            cam = cam.view(self.num_classes, fmap_size, -1)
        features = self.gap(features).view(batch_size, self.in_features)
        logits = self.fc(features)
        target = logits.new_full(logits.shape, 0)
        if im_labels is not None:
            for i in range(batch_size):
                one_hot_labels = self.mlb.transform(im_labels[i].view(1, -1).numpy())
                target[i] = torch.from_numpy(one_hot_labels)
        target = target.to(im_data.device)
        if self.training:
            return logits, target
        else:
            return logits, target, cam
