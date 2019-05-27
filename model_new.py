import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from correlation_package.correlation import Correlation


def model_initial(object):
    for m in object.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class SamePad2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FPN(nn.Module):
    def __init__(self, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.inplanes = 32

        self.C1_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.C1_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.AOD = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.C2 = self.make_layer(32, 3, stride=2)
        self.C3 = self.make_layer(64, 3, stride=2)
        self.C4 = self.make_layer(96, 3, stride=2)
        self.C5 = self.make_layer(128, 3, stride=2)
        self.C6 = self.make_layer(160, 3, stride=2)
        self.C7 = self.make_layer(192, 3, stride=2)

        self.P7_conv1 = nn.Conv2d(192*4, self.out_channels, kernel_size=1, stride=1)
        self.P7_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1)
        )
        self.P6_conv1 = nn.Conv2d(160*4, self.out_channels, kernel_size=1, stride=1)
        self.P6_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1)
        )
        self.P5_conv1 = nn.Conv2d(128*4, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1)
        )
        self.P4_conv1 =  nn.Conv2d(96*4, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1)
        )
        self.P3_conv1 = nn.Conv2d(64*4, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1)
        )
        self.P2_conv1 = nn.Conv2d(32*4, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1)
        )

    def forward(self, x):
        x = self.C1_1(x)
        trans = torch.cat((self.C1_2(x), x), 1)
        trans = self.AOD(trans)
        x = x * trans

        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        c5_out = x
        x = self.C6(x)
        c6_out = x
        x = self.C7(x)

        p7_out = self.P7_conv1(x)
        p6_out = self.P6_conv1(c6_out) + F.upsample(p7_out, scale_factor=2, mode='bilinear')
        p5_out = self.P5_conv1(c5_out) + F.upsample(p6_out, scale_factor=2, mode='bilinear')
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2, mode='bilinear')
        p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2, mode='bilinear')
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2, mode='bilinear')

        p7_out = self.P7_conv2(p7_out)
        p6_out = self.P6_conv2(p6_out)
        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out, p7_out]

    def make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes*4, kernel_size=1, stride=stride
                    ),
                nn.BatchNorm2d(
                    planes * 4, eps=0.001, momentum=0.01
                    )
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

class FlowNet(nn.Module):
    def __init__(self, md = 4):
        super(FlowNet, self).__init__()
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = nn.Conv2d(od+dd[4], 2, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = nn.ConvTranspose2d(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 224 + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = nn.Conv2d(od+dd[4], 2, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = nn.ConvTranspose2d(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 224 + 4
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = nn.Conv2d(od+dd[4], 2, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = nn.ConvTranspose2d(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 224 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = nn.Conv2d(od+dd[4], 2, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = nn.ConvTranspose2d(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 224 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = nn.Conv2d(od+dd[4], 2, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

    def warp(self, x, flow):
        B, C, H, W = x.size()

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flow


        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, x1, x2):
        c11, c12, c13, c14, c15, c16 = x1
        c21, c22, c23, c24, c25, c26 = x2

        corr6 = self.corr(c16, c26)
        corr6 = self.relu(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = self.relu(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.relu(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.relu(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.relu(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        flow1 = F.upsample(flow2, scale_factor=4, mode='bilinear')

        return flow1, [flow2, flow3, flow4, flow5, flow6]

class OpticalFlow(nn.Module):
    def __init__(self):
        super(OpticalFlow, self).__init__()
        self.fpn = FPN(out_channels=224)
        self.flownet = FlowNet()

    def forward(self, x1, x2):
        x1 = self.fpn(x1)
        x2 = self.fpn(x2)
        flow, output = self.flownet(x1, x2)

        return flow, output