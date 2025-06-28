import numpy as np 
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.util import get_model
from copy import deepcopy
from typing import Tuple, List
import torch.nn.functional as F
from models.d2s import AttentionalGNN

class L2S(BaseModel):
    default_conf = {
        'trainable': True,
        'n_heads': 4,
        'd_inner': 1024,
        'n_att_layers': 1,
        'feature_dim': 256,
        'GNN_no_layers': 1,
        'keeping_threshold': 0.01,
        'mapping_layers': [512, 1024, 512],
    }
    required_data = ['lines_descriptors']

    def _init(self, conf):
        self.line_encoder = LineEncoder(conf.feature_dim, conf.n_heads, conf.n_att_layers, conf.d_inner)
        self.gnn1 = AttentionalGNN(
            feature_dim=self.conf.feature_dim, no_layers=1)
        # self.gnn_rest = AttentionalGNN(
        #     feature_dim=self.conf.feature_dim, no_layers=self.conf.GNN_no_layers)
        
        self.predict_mask = MLP([conf.feature_dim]+[256]+[1])
        self.mapping_l = MLP([conf.feature_dim]+self.conf.mapping_layers+[6])  # mapping line descriptors to 3D lines
        
    def _forward(self, data):
        # get line descriptors
        l_desc = self.line_encoder(data['lines_descriptors'])
        l_desc = self.gnn1(l_desc)
        # Predict unimportant lines
        prd_mask_coarse = self.predict_mask(l_desc)
        # prd_mask = torch.sigmoid(prd_mask)
        pred = {}
        pred['prd_mask_lines'] = None
        if self.training:
            gt_mask = data['validLines']
            l_desc = l_desc[:,:,gt_mask.squeeze() > 0]
        else:
            prd_mask = torch.sigmoid(prd_mask_coarse)
            prd_mask = prd_mask > self.conf.keeping_threshold
            prd_mask = prd_mask[:,0,:]
            l_desc = l_desc[:,:,prd_mask[0,:]]
            pred['prd_mask_lines'] = prd_mask
            
        if l_desc.size(2) != 0:
            # l_desc = self.gnn_rest(l_desc)
            pred['lines3D'] = self.mapping_l(l_desc)
        else:
            pred['lines3D'] = None
        pred['prd_mask_lines_coarse'] = prd_mask_coarse
        return pred
    
    def loss(self, pred, data):
        pass


class ScaledDotProduct(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale, attn_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.scale, k.transpose(3, 4))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention_Line(nn.Module):
    """ Multi-Headed Attention """
    def __init__(self, n_heads: int, d_feature: int, dropout=0.1):
        super().__init__()
        assert d_feature % n_heads == 0
        dim = d_feature // n_heads
        self.dim = dim
        self.n_heads = n_heads

        self.w_qs = nn.Linear(d_feature, n_heads * dim, bias=True)
        self.w_ks = nn.Linear(d_feature, n_heads * dim, bias=True)
        self.w_vs = nn.Linear(d_feature, n_heads * dim, bias=True)
        self.fc = nn.Linear(n_heads * dim, d_feature, bias=True) 

        self.attention = ScaledDotProduct(scale = dim ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_feature, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k = self.dim
        d_v = self.dim
        n_heads = self.n_heads

        n_batches = q.size(0)
        n_sublines = q.size(1)
        n_words_q = q.size(2)
        n_words_k = k.size(2)
        n_words_v = v.size(2)

        residual = q

        q = self.w_qs(q).view(n_batches, n_sublines, n_words_q, n_heads, d_k)
        k = self.w_ks(k).view(n_batches, n_sublines, n_words_k, n_heads, d_k)
        v = self.w_vs(v).view(n_batches, n_sublines, n_words_v, n_heads, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        if mask is not None:
            mask = mask.unsqueeze(2)   # For head axis broadcasting.
            
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(2,3).contiguous().view(n_batches, n_sublines, n_words_q, -1)
        q = self.dropout(self.fc(q))
        
        q += residual
        q = self.layer_norm(q)

        return q, attn

class FeedForward(nn.Module):
    """ Feed Forward layer """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)   # d_in: 256, d_hid: 1024
        self.w_2 = nn.Linear(d_hid, d_in)   # d_hid: 1024, d_in: 256
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class LineDescriptiveEncoder(nn.Module):
    """ Line Descriptive Network using the transformer """
    def __init__(self, d_feature: int, n_heads: int, d_inner: int, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention_Line(n_heads, d_feature)
        self.pos_ffn = FeedForward(d_feature, d_inner, dropout=dropout)

    def forward(self, desc, slf_attn_mask=None):

        desc, enc_slf_attn = self.slf_attn(desc, desc, desc, mask=slf_attn_mask)
        desc = self.pos_ffn(desc)

        return desc, enc_slf_attn

class LineEncoder(nn.Module):
    """ LineEncoder mimics the transformer model"""
    def __init__(self, feature_dim, n_heads, n_att_layers, d_inner, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.desc_layers = nn.ModuleList([
            LineDescriptiveEncoder(feature_dim, n_heads, d_inner, dropout=dropout) 
            for _ in range(n_att_layers)])

    def forward(self, desc, return_attns=False):
        enc_slf_attn_list = []
        for desc_layer in self.desc_layers:
            enc_output, enc_slf_attn = desc_layer(desc)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        # get the first token of each line
        sentence =  enc_output[:,:,0,:].transpose(1,2)
        return sentence # line descriptors


def MLP(channels:list):
    layers = []
    n_chnls = len(channels)
    for i in range(1, n_chnls):
        layers.append(nn.Conv1d(channels[i-1], channels[i], 
                                kernel_size=1, bias=True))
        if i < n_chnls-1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)





