import torch
from torch import nn



def conv1d(param, adaptation, meta):
    w = nn.Parameter(torch.ones(param['out_channels'], param['in_channels'], param['kernel_size']))
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(param['out_channels']))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b

def linear(param, adaptation, meta):
    w = nn.Parameter(torch.ones(param['out'], param['in']))
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(param['out']))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b


def attention(param, adaptation, meta):
    w_k, b_k = linear(param, adaptation, meta)
    w_q, b_q = linear(param, adaptation, meta)
    w_v, b_v = linear(param, adaptation, meta)
    return [w_k, w_q, w_v], [b_k, b_q, b_v]


def bn(param, adaptation, meta):
    w = nn.Parameter(torch.ones(param['in_channels']))
    
    b = nn.Parameter(torch.ones(param['in_channels']))
    m = nn.Parameter(torch.zeros(param['in_channels']), requires_grad=False)
    v = nn.Parameter(torch.zeros(param['in_channels']), requires_grad=False)
    return w, b, m, v