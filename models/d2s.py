# This file is part of D2S paper
# "D2S": https://arxiv.org/abs/2307.15250
from torch import nn
import torch.nn.functional as F
import torch
from typing import Tuple
from copy import deepcopy
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel


class D2S(BaseModel):
    default_conf = {
        'feature_dim': 256,
        'GNN_no_layers': 5,
        'trainable': True,
        'keeping_threshold': 0.8,
        'mapping_layers': [512, 1024, 1024, 512],
    }
    required_data = ['points_descriptors']
    def _init(self, conf):
        # self.gnn1 = AttentionalGNN(
        #     feature_dim=conf.feature_dim, no_layers=1)
        self.gnn_rest = AttentionalGNN(
            feature_dim=conf.feature_dim, no_layers=self.conf.GNN_no_layers)
        
        self.predict_mask = MLP([conf.feature_dim]+[256]+[1])
        self.mapping_p = MLP([conf.feature_dim]+self.conf.mapping_layers+[3])

    def _forward(self, data): # this is for training
        descpt = data['points_descriptors']
        # out = self.gnn1(descpt)
        # Predict unimportant points
        prd_mask_coarse = self.predict_mask(descpt)
        # prd_mask = torch.sigmoid(prd_mask)
        # pruning unimportant points3D
        pred = {}
        pred['prd_mask_points'] = None
        if self.training:
            gt_mask = data['validPoints']
            out = descpt[:,:,gt_mask.squeeze() > 0]
        else:
            # prd_mask[:,1,:] = 1/(1+100*torch.abs(prd_mask[:,1,:]))
            if 'validPoints' in data.keys():
                gt_mask = data['validPoints']
                prd_mask = gt_mask > 0
                out = descpt[:,:,prd_mask.squeeze()]
            else:
                prd_mask = torch.sigmoid(prd_mask_coarse)
                prd_mask = prd_mask > self.conf.keeping_threshold
                prd_mask = prd_mask[:,0,:]
                out = descpt[:,:,prd_mask[0,:]]
            pred['prd_mask_points'] = prd_mask
        out = self.gnn_rest(out)
        pred['points3D'] = self.mapping_p(out)
        pred['prd_mask_points_coarse'] = prd_mask_coarse
        return pred
    
    def loss(self, pred, data):
        raise NotImplementedError   

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
    def __init__(self, feature_dim: int, no_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(no_layers)])
        self.no_layers = no_layers

    def forward(self, desc: torch.Tensor):
        for i in range(self.no_layers):
            delta = self.layers[i](desc, desc)
            desc = desc + delta
        return desc