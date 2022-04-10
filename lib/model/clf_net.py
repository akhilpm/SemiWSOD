from torchvision.models import vgg16
from torchvision.models import resnet50
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from config import cfg

class Cls_Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([np.arange(1, num_classes+1)])

        self.cls_net = vgg16(pretrained=True)
        self.cls_net.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        #self.cls_net = resnet50(pretrained=True)
        #self.cls_net.fc = nn.Linear(in_features=2048, out_features=num_classes)


    def forward(self, im_data, im_labels=None):
        batch_size = im_data.size(0)
        logits = self.cls_net(im_data)
        target = logits.new_full(logits.shape, 0)
        if im_labels is not None:
            for i in range(batch_size):
                one_hot_labels = self.mlb.transform(im_labels[i].view(1, -1).numpy())
                target[i] = torch.from_numpy(one_hot_labels)
        target = target.to(im_data.device)
        return logits, target
