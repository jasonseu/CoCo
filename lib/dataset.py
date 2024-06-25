# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-7-15
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import logging
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from randaugment import RandAugment
from .cutout import CutoutPIL, SLCutoutPIL


logger = logging.getLogger(__name__)


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MLDataset(Dataset):
    def __init__(self, cfg, data_path, training=True):
        super(MLDataset, self).__init__()
        self.cfg = cfg
        self.labels = [line.strip() for line in open(cfg.label_path)]
        self.num_classes = len(self.labels)
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.data = []
        with open(data_path, 'r') as fr:
            for line in fr.readlines():
                img_path, img_label = line.strip().split('\t')
                img_label = [self.label2id[l] for l in img_label.split(',')]
                self.data.append([img_path, img_label])
        
        self.transform = self.get_transform(training) 
        logger.info(self.transform)

    def get_transform(self, training):
        t = []
        t.append(transforms.Resize((self.cfg.img_size, self.cfg.img_size)))
        if training:
            t.append(transforms.RandomHorizontalFlip())
            t.append(CutoutPIL())
            t.append(RandAugment())
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return transforms.Compose(t)

    def __getitem__(self, index):
        img_path, img_label = self.data[index]
        img_data = Image.open(img_path).convert('RGB')
        img_data = self.transform(img_data)

        # one-hot encoding for label
        target = np.zeros(self.num_classes).astype(np.float32)
        target[img_label] = 1.0
        
        item = {
            'img': img_data,
            'target': target,
            'img_path': img_path
        }
        
        return item
        
        
    def __len__(self):
        return len(self.data)