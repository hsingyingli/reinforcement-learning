import torch
from torch import nn
from torch.nn import functional as F

import layers


class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config
        self.vars, self.vars_bn = self.parse_config()


    def parse_config(self):
        vars_list = nn.ParameterList()
        vars_bn = nn.ParameterList()

        for i, info_dict in enumerate(self.config):
            if info_dict['name'] == 'conv1d':
                w, b = layers.conv1d(info_dict['config'], info_dict['adaptation'], info_dict['meta'])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict['name'] == 'linear':
                w, b = layers.linear(info_dict['config'], info_dict['adaptation'],
                                     info_dict['meta'])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict['name'] == 'attention':
                w, b = layers.attention(info_dict['config'], info_dict['adaptation'], info_dict['meta'])
                for wi, bi in zip(w, b):
                    vars_list.append(wi)
                    vars_list.append(bi)
                    
            elif info_dict['name'] == 'bn':
                w, b, m, v = layers.bn(info_dict['config'], info_dict['adaptation'], info_dict['meta'])
                vars_list.append(w)
                vars_list.append(b)
                vars_bn.extend([m, v])

            elif info_dict['name'] in ['tanh', 'rep', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                       'flatten', 'reshape', 'padding_reshape', 'sigmoid', 'rotate']:
                continue
        
        return vars_list, vars_bn

    def reset_vars(self):
        for var in self.vars:
            if var.adaptation is True:
                if len(var.shape) > 1:
                    torch.nn.init.kaiming_normal_(var)
                else:
                    torch.nn.init.zeros_(var)

    def forward(self, x, vars=None, vars_bn = None, config=None, sparsity_log=False, rep=False, bn_training = True):
        if vars is None:
            vars = self.vars

        if vars_bn is None:
            vars_bn = self.vars_bn

        if config is None:
            config = self.config
        
        idx = 0
        bn_idx = 0

        for info_dict in config:
            name  = info_dict['name']

            if name == 'conv1d':
                w, b = vars[idx], vars[idx + 1] 
                x = F.conv1d(x, w, b, stride=info_dict['config']['stride'], padding=info_dict['config']['padding'])
                idx += 2

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'attention':
                w_k, b_k = vars[idx], vars[idx + 1]
                w_q, b_q = vars[idx+2], vars[idx + 3]
                w_v, b_v = vars[idx+4], vars[idx + 5]

                key = F.linear(x, w_k, b_k)
                query = F.linear(x, w_q, b_q)
                value = F.linear(x, w_v, b_v)

                score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
                p_attn = F.softmax(score, dim = -1)

                x = torch.matmul(p_attn, value)

                idx += 6

            elif name == 'bn':
                w, b = vars[idx], vars[idx+1]
                m, v = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                out = F.batch_norm(out, m, v, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'relu':
                x = F.relu(x)
            elif name == 'flatten':
                x = x[:,:,-1]
            elif name == 'padding_reshape':
                x = x[:,:,1:]

        assert idx == len(vars)
        return x

    def update_weights(self, vars):

        for old, new in zip(self.vars, vars):
            old.data = new.data

    def get_adaptation_parameters(self, vars=None):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        if vars is None:
            vars = self.vars
        return list(filter(lambda x: x.adaptation, list(vars)))
    
    def get_forward_meta_parameters(self):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        return list(filter(lambda x: x.meta, list(self.vars)))

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """

        return self.vars
        