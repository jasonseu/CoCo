# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-9-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import copy
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv


class LowRankBilinearAttention(nn.Module):
    """
    Low-rank bilinear attention network.
    """
    def __init__(self, dim1, dim2, att_dim=2048):
        """
        :param dim1: feature size of encoded images
        :param dim2: feature size of encoded labels
        :param att_dim: size of the attention network
        """
        super().__init__()
        self.linear1 = nn.Linear(dim1, att_dim, bias=False)  # linear layer to transform encoded image
        self.linear2 = nn.Linear(dim2, att_dim, bias=False)  # linear layer to transform decoder's output
        self.hidden_linear = nn.Linear(att_dim, att_dim)   # linear layer to calculate values to be softmax-ed
        self.target_linear = nn.Linear(att_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)  # softmax layer to calculate weights

    def forward(self, x1, x2):
        """
        Forward propagation.
        :param 
            x1: a tensor of dimension (B, num_pixels, dim1)
            x2: a tensor of dimension (B, num_labels, dim2)
        """
        _x1 = self.linear1(x1).unsqueeze(dim=1)  # (B, 1, num_pixels, att_dim)
        _x2 = self.linear2(x2).unsqueeze(dim=2)  # (B, num_labels, 1, att_dim)
        t = self.hidden_linear(self.tanh(_x1 * _x2))
        t = self.target_linear(t).squeeze(-1)  # B, num_labels, num_pixels
        alpha = self.softmax(t)  # (B, num_labels, num_pixels)
        label_repr = torch.bmm(alpha, x1)
        return label_repr, alpha


class GatedGNN(nn.Module):
    def __init__(self, dim, num_steps):
        super().__init__()
        self.fc1_w = nn.Linear(dim, dim, bias=False)
        self.fc1_u = nn.Linear(dim, dim, bias=False)
        self.fc2_w = nn.Linear(dim, dim, bias=False)
        self.fc2_u = nn.Linear(dim, dim, bias=False)
        self.fc3_w = nn.Linear(dim, dim, bias=False)
        self.fc3_u = nn.Linear(dim, dim, bias=False)
        self.num_steps = num_steps

    def forward(self, feats, matrix):
        feat_list = [feats]
        for _ in range(self.num_steps):
            infeats = torch.bmm(matrix, feats)
            feats = self.update(infeats, feats)
            feat_list.append(feats)

        return feats, feat_list

    def update(self, x, h):
        t1 = torch.sigmoid(self.fc1_w(x) + self.fc1_u(h))
        t2 = torch.sigmoid(self.fc2_w(x) + self.fc2_u(h))
        t3 = torch.tanh(self.fc3_w(x) + self.fc3_u(t2 * h))
        t4 = (1 - t1) * h + t1 * t3

        return t4


class GNN(nn.Module):
    def __init__(self, dim, num_steps):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, feats, matrix):
        feat_list = [feats]
        for _ in range(self.num_steps):
            infeats = torch.bmm(matrix, feats)
            feats = self.update(infeats, feats)
            feat_list.append(feats)

        return feats, feat_list

    def update(self, x, h):
        t4 = 0.5 * x + 0.5 * h
        return t4
    
    
class GCN(nn.Module):
    def __init__(self, dim, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.weight = nn.Conv1d(dim, dim, 1, bias=False)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, feats, matrix):
        feats = torch.transpose(feats, dim0=1, dim1=2)
        matrix = torch.transpose(matrix, dim0=1, dim1=2)
        for i in range(self.num_steps):
            temp = torch.bmm(feats, matrix)
            temp = self.relu(temp)
            feats = self.weight(temp)
            feats = self.relu(feats)
        feats = torch.transpose(feats, dim0=1, dim1=2)
        return feats, []

    
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, activation, k: int = 2):
#         super(Encoder, self).__init__()
#         assert k >= 2
#         self.k = k
#         self.conv = [GCNConv(in_channels, 2 * out_channels)]
#         for _ in range(1, k-1):
#             self.conv.append(GCNConv(2 * out_channels, 2 * out_channels))
#         self.conv.append(GCNConv(2 * out_channels, out_channels))
#         self.conv = nn.ModuleList(self.conv)
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'leakyrelu':
#             self.activation = nn.LeakyReLU()

#     def forward(self, x, edge_index, edge_weights):
#         for i in range(self.k):
#             x = self.activation(self.conv[i](x, edge_index, edge_weights))
#         return x


class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x, 2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class JSDivergence(nn.Module):
    def __init__(self):
        super(JSDivergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p, q):
        # p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        js_div = 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
        js_div = js_div.sum(-1)
        n = js_div.shape[0]
        js_div = js_div.sum() / (n * (n-1))
        return js_div
    
    
    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0),1,1,1))


def build_position_encoding(hidden_dim, arch, position_embedding, img_size):
    N_steps = hidden_dim // 2

    if arch in ['CvT_w24'] or 'vit' in arch:
        downsample_ratio = 16
    else:
        downsample_ratio = 32

    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        assert img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(img_size)
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=img_size // downsample_ratio, maxW=img_size // downsample_ratio)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding

    
class TransformerEncoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderLayer(d_model, nhead)) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    

class TransformerDecoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderLayer(d_model, nhead)) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2, sim_mat_2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu") -> None:
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers)

        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers)
        
    def forward(self, src, query_embed, pos_embed, mask=None):
        # src = src.permute(2, 0, 1) 
        batch_size = src.shape[1]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        if mask is not None:
            mask = mask.flatten(1)

        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2)

