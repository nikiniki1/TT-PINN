#!/usr/bin/env python3

import sys 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim

from tc.tc_cores import TensorTrain, _are_tt_cores_valid
import tc.tc_math 
from tc.tc_init import get_variables, glorot_initializer, he_initializer, lecun_initializer
import tc.tc_decomp

from torch.utils.data import TensorDataset, DataLoader 

activations = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear']
inits = ['glorot', 'he', 'lecun']

class TTLinear(nn.Module):
    """A Tensor-Train linear layer is implemented by Pytorch in this calss. The 
    Tensor-Train linear layer replaces a fully-connected one by factorizing 
    it into a smaller 4D tensors and reducing the total number of parameters 
    of the dense layer. So the training and inference process can be significantly 
    sped up without a loss of model performance. That's particularly important 
    for the tasks of feature selection or dimension reduction where the information 
    redudancy always exists. """
    def __init__(self, inp_modes, out_modes, tt_rank, init='glorot',
                bias_init=0.1, activation='relu', **kwargs):
        super(TTLinear, self).__init__()
        self.ndims = len(inp_modes)
        self.inp_modes = inp_modes 
        self.out_modes = out_modes 

        self.tt_shape = [inp_modes, out_modes]
        self.activation = activation 
        self.init = init 
        self.tt_rank = tt_rank

        if self.init == 'glorot':
            initializer = glorot_initializer(self.tt_shape, tt_rank=tt_rank)
        elif self.init == 'he':
            initializer = he_initializer(self.tt_shape, tt_rank=tt_rank)
        elif self.init == 'lecun':
            initializer = lecun_initializer(self.tt_shape, tt_rank=tt_rank)
        else:
            raise ValueError('Unknown init "%s", only %s are supported'%(self.init, inits))

        self.W_cores = get_variables(initializer)
        _are_tt_cores_valid(self.W_cores, self.tt_shape, self.tt_rank)
        self.b = torch.nn.Parameter(torch.ones(1))

    
    def forward(self, x):
        TensorTrain_W = TensorTrain(self.W_cores, self.tt_shape, self.tt_rank)
        h = tc.tc_math.matmul(x, TensorTrain_W)
        if self.activation is not None:
             if self.activation in activations:
                 if self.activation == 'sigmoid':
                     h = torch.sigmoid(h)
                 elif self.activation == 'tanh':
                     h = torch.tanh(h)
                 elif self.activation == 'relu':
                     h = torch.relu(h)
                 elif self.activation == 'linear':
                     h = h 
             else:
                 raise ValueError('Unknown activation "%s", only %s and None \
                    are supported'%(self.activation, activations))

        return h

    def extra_repr(self):
        return '(TTLayer): inp_modes={}, out_modes={}, mat_ranks={}'.format(
                list(self.inp_modes), \
                list(self.out_modes), \
                list(self.tt_rank))

    def nelements(self):
        """
        Returns the number of parameters of TTLayer.
        """
        num = 0
        for i in range(self.ndims):
            num += self.inp_modes[i] * self.out_modes[i] * \
                self.tt_rank[i] * self.tt_rank[i+1] 
        num = num + np.prod(self.out_modes)

        return num

    @property
    def name(self):
        return "TTLinear"