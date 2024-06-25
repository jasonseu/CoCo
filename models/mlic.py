# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-9-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn

from .utils import *
from .factory import register_model, create_backbone
from lib.utils import get_loss_fn


__all__ = ['mlic']


class MLIC(nn.Module):
    def __init__(self, backbone, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.query_embed = nn.Embedding(cfg.num_classes, feat_dim)
        self.attention = nn.Transformer(
            d_model=feat_dim, 
            nhead=cfg.num_heads, 
            num_decoder_layers=cfg.enc_layers, 
            num_encoder_layers=cfg.dec_layers
        )
        self.net = nn.Linear(feat_dim * 2, cfg.num_classes)
        self.encoder = GatedGNN(feat_dim, cfg.num_steps)
        self.fc = Element_Wise_Layer(cfg.num_classes, feat_dim * 2)
        self.criterion = get_loss_fn(cfg)
        
    def forward(self, x, y=None):
        x = self.backbone(x)
        x_pool = torch.mean(x, dim=1)

        query_input = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)
        x = x.permute(1, 0, 2)
        x = self.attention(x, query_input)
        x = x.permute(1, 0, 2)

        x_pool = x_pool.unsqueeze(1).repeat(1, x.shape[1], 1)
        _x = torch.cat([x, x_pool], dim=-1)
        logits = self.net(_x)
        weights = torch.sigmoid(logits)

        ax, feat_list = self.encoder(x, weights)
        _x = torch.cat([x, ax], dim=-1)
        logits = self.fc(_x)
        
        loss = bce = ctr = 0.0
        if self.training:
            if self.cfg.lamda != 0:
                masks = torch.bmm(y.unsqueeze(-1), y.unsqueeze(1))
                posi_weights = weights * masks        # yield positive view, randomly drop negative edges
                nega_weights = weights * (1 - masks)  # yield negative view, randomly drop positive edges
                with torch.no_grad():
                    px, _ = self.encoder(x, posi_weights)
                    nx, _ = self.encoder(x, nega_weights)
            
                for i in range(x.shape[0]):
                    _y = y[i]
                    p_ax = ax[i][_y == 1, :]
                    n_ax = ax[i][_y == 0, :]
                    p_px = px[i][_y == 1, :]
                    p_nx = nx[i][_y == 1, :]
                    # t = torch.sum(p_ax * p_px, dim=-1)
                    app_logits = torch.sum(p_ax * p_px, dim=-1) / self.cfg.tau  # anchor view & positive view
                    ctr_logits = app_logits.unsqueeze(-1)
                    
                    anp_logits = torch.sum(p_ax * p_nx, dim=-1) / self.cfg.tau  # anchor view & negative view
                    ctr_logits = torch.cat([ctr_logits, anp_logits.unsqueeze(-1)], dim=-1)
                    
                    aap_logits = torch.mm(p_ax, p_ax.transpose(0, 1)) / self.cfg.tau  # positive sample & other positive sample in anchor view
                    n = aap_logits.shape[0]
                    aap_logits = aap_logits.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)  # remove diagonal elements
                    ctr_logits = torch.cat([ctr_logits, aap_logits], dim=-1)

                    targets = torch.tensor([0] * ctr_logits.shape[0], dtype=torch.long).cuda()
                    nce = nn.CrossEntropyLoss()(ctr_logits, targets)
                    ctr += nce
                    
                ctr /= x.shape[0]

            bce = self.criterion(logits, y)
            
        return {
            'logits': logits,
            'bce': bce,
            'ctr': ctr,
            'ax': ax,
            'feat_list': feat_list
        }
        

@register_model
def mlic(cfg):
    backbone, feat_dim = create_backbone(cfg.arch, img_size=cfg.img_size)
    model = MLIC(backbone, feat_dim, cfg)
    return model
