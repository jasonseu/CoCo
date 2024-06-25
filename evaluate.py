# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-11-11
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.factory import create_model
from lib.metrics import *
from lib.dataset import MLDataset

torch.backends.cudnn.benchmark = True


class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        dataset = MLDataset(cfg, cfg.test_path, training=False)
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        self.labels = dataset.labels
        
        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda()

        self.cfg = cfg
        self.voc12_mAP = VOCmAP(cfg.num_classes, year='2012', ignore_path=cfg.ignore_path)
        self.topk_meter = TopkMeter(cfg.num_classes, cfg.ignore_path, threshold=cfg.threshold, topk=cfg.topk)
        self.threshold_meter = ThresholdMeter(cfg.num_classes, cfg.ignore_path, threshold=cfg.threshold)

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_ema_best_path)
        if list(model_dict.keys())[0].startswith('module'):
            model_dict = {k[7:]: v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)
        print('loading best checkpoint success')
        
        self.model.eval()
        self.voc12_mAP.reset()
        self.topk_meter.reset()
        self.threshold_meter.reset()
        for batch in tqdm(self.dataloader):
            img = batch['img'].cuda()
            targets = batch['target'].cuda()
            ret = self.model(img, targets)

            logit = ret['logits']
            scores = torch.sigmoid(logit).cpu().numpy()
            targets = targets.cpu().numpy()
            self.voc12_mAP.update(scores, targets)
            self.topk_meter.update(scores, targets)
            self.threshold_meter.update(scores, targets)

        _, mAP = self.voc12_mAP.compute()
        self.topk_meter.compute()
        self.threshold_meter.compute()

        print('model {} data {} mAP: {:.4f}'.format(self.cfg.model, self.cfg.data, mAP))
        print('Overall metrics: CP: {:4f}, CR: {:4f}, CF1: {:4F}'.format(self.threshold_meter.cp, self.threshold_meter.cr, self.threshold_meter.cf1))
        print('                 OP: {:4f}, OR: {:4f}, OF1: {:4F}'.format(self.threshold_meter.op, self.threshold_meter.or_, self.threshold_meter.of1))
        print('Topk metrics: CP: {:4f}, CR: {:4f}, CF1: {:4F}'.format(self.topk_meter.cp, self.topk_meter.cr, self.topk_meter.cf1))
        print('              OP: {:4f}, OR: {:4f}, OF1: {:4F}'.format(self.topk_meter.op, self.topk_meter.or_, self.topk_meter.of1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='experiments/mlic11_voc2007/exp1')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r'))
    cfg = Namespace(**cfg)
    cfg.batch_size = args.batch_size
    cfg.threshold = args.threshold
    print(cfg)
    
    evaluator = Evaluator(cfg)
    evaluator.run()
