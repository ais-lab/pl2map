import numpy as np 
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.util import get_model
from copy import deepcopy
from typing import Tuple, List
import torch.nn.functional as F

class PL2Map(BaseModel):
    default_conf = {
        'trainable': True,
        'n_heads': 4,
        'd_inner': 1024,
        'n_att_layers': 1,
        'feature_dim': 256,
        'GNN_layers': ['self', 'cross', 'self', 'cross', 'self'],
        'mapping_layers': [512, 1024, 512],
    }
    required_data = ['points_descriptors', 'lines_descriptors']

    def _init(self, conf):
        self.line_encoder = LineEncoder(conf.feature_dim, conf.n_heads, conf.n_att_layers, conf.d_inner)
        self.gnn = AttentionalGNN(
            feature_dim=self.conf.feature_dim, layer_names=self.conf.GNN_layers)
        self.mapping_p = MLP([conf.feature_dim]+self.conf.mapping_layers+[4])  # mapping point descriptors to 3D points
        self.mapping_l = MLP([conf.feature_dim]+self.conf.mapping_layers+[7])  # mapping line descriptors to 3D lines
        
    def _forward(self, data):
        # get line descriptors
        p_desc = data['points_descriptors']
        l_desc = self.line_encoder(data['lines_descriptors'])
        p_desc, l_desc = self.gnn(p_desc, l_desc)
        pred = {}
        pred['points3D'] = self.mapping_p(p_desc)
        pred['lines3D'] = self.mapping_l(l_desc)
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


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
    
    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1