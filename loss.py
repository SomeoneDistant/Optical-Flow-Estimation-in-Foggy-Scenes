import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os


# __all__ = [
#     'MultiScale'
# ]

# def EPE(input_flow, target_flow, mean=True):
#     batch_size = input_flow.shape[0]
#     if mean:
#         return torch.norm(target_flow-input_flow, p=2, dim=1).mean()
#     else:
#         return torch.norm(target_flow-input_flow, p=2, dim=1).sum()/batch_size

# class MultiScale(nn.Module):
#     def __init__(self, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L2'):
#         super(MultiScale, self).__init__()

#         self.startScale = startScale
#         self.numScales = numScales
#         # self.loss_weights = torch.FloatTensor([l_weight / 2 ** scale for scale in range(self.numScales)]).cuda()
#         self.loss_weights = torch.FloatTensor([0.005, 0.01, 0.02, 0.08, 0.32]).cuda()
#         self.l_type = norm
#         self.div_flow = 0.05

#         assert(len(self.loss_weights) == self.numScales)

#         self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]

#     def forward(self, flow, output, target):
#         lossvalue = 0
#         # if type(output) is tuple:
#         #     target = self.div_flow * target
#         #     for i, output_ in enumerate(output):
#         #         target_ = self.multiScales[i](target)
#         #         epevalue += self.loss_weights[i]*EPE(output_, target_)
#         #         lossvalue += self.loss_weights[i]*self.loss(output_, target_)
#         #     return [lossvalue, epevalue]
#         # else:
#         #     epevalue += EPE(output, target)
#         #     lossvalue += self.loss(output, target)
#         #     return  [lossvalue, epevalue]

#         # target = self.div_flow * target
#         for i, output_ in enumerate(output):
#             target_ = self.multiScales[i](target)
#             lossvalue += self.loss_weights[i] * EPE(output_, target_, mean=False)
#         epevalue = EPE(target, flow)
#         return [lossvalue, epevalue]


def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow.cuda()-input_flow.cuda(),2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()/batch_size
    else:
        return EPE_map.sum()/batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.upsample(target, (h, w), mode='bilinear')
        return EPE(output, target_scaled, sparse, mean=False)
        #return nn.MSELoss()(output, target_scaled)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.upsample(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)