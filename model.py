import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from correlation_package.correlation import Correlation
import cv2

import os


__all__ = [
    'Extractor', 'FPN_Extractor', 'Discriminator', 'Generator', 'model_initial', 'PWCNet'
]


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
        nn.LeakyReLU(0.1)
        )

def conv_norm(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)

        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)

        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)

        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)

        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)

        self.conv6a = conv(128, 196, kernel_size=3, stride=2)
        self.conv6aa = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)

    def forward(self, im1, im2):
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))

        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))

        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))

        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))

        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))

        c16 = self.conv6b(self.conv6aa(self.conv6a(c15)))
        c26 = self.conv6b(self.conv6aa(self.conv6a(c25)))

        return [c11, c21, c12, c22, c13, c23, c14, c24, c15, c25, c16, c26]

class Basicblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Basicblock, self).__init__()
        self.conv1 = conv_norm(in_planes, out_planes, kernel_size=3, stride=2)
        self.conv2 = conv_norm(out_planes, out_planes, kernel_size=3)
        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.leakyRELU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        output = self.bn(self.conv3(self.conv2(x)))
        output += x
        output = self.leakyRELU(output)
        return output

class FPN_Extractor(nn.Module):
    def __init__(self):
        super(FPN_Extractor, self).__init__()
        self.layer1 = Basicblock(3, 16)
        self.layer2 = Basicblock(16, 32)
        self.layer3 = Basicblock(32, 64)
        self.layer4 = Basicblock(64, 96)
        self.layer5 = Basicblock(96, 128)
        self.layer6 = Basicblock(128, 196)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(96, 64, kernel_size=1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(128, 96, kernel_size=1, stride=1, bias=False)
        self.conv6 = nn.Conv2d(196, 128, kernel_size=1, stride=1, bias=False)

        self.smooth1 = conv_norm(16, 16, kernel_size=3, stride=1, padding=1)
        self.smooth2 = conv_norm(32, 32, kernel_size=3, stride=1, padding=1)
        self.smooth3 = conv_norm(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth4 = conv_norm(96, 96, kernel_size=3, stride=1, padding=1)
        self.smooth5 = conv_norm(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, im1, im2):
        c11 = self.layer1(im1)
        c12 = self.layer2(c11)
        c13 = self.layer3(c12)
        c14 = self.layer4(c13)
        c15 = self.layer5(c14)
        c16 = self.layer6(c15)

        c21 = self.layer1(im2)
        c22 = self.layer2(c21)
        c23 = self.layer3(c22)
        c24 = self.layer4(c23)
        c25 = self.layer5(c24)
        c26 = self.layer6(c25)

        # c15 = torch.cat((c15, F.upsample_bilinear(self.conv6(c16), scale_factor=2)), 1)
        # c15 = self.smooth5(c15)
        # c14 = torch.cat((c14, F.upsample_bilinear(self.conv5(c15), scale_factor=2)), 1)
        # c14 = self.smooth4(c14)
        # c13 = torch.cat((c13, F.upsample_bilinear(self.conv4(c14), scale_factor=2)), 1)
        # c13 = self.smooth3(c13)
        # c12 = torch.cat((c12, F.upsample_bilinear(self.conv3(c13), scale_factor=2)), 1)
        # c12 = self.smooth2(c12)
        # c11 = torch.cat((c11, F.upsample_bilinear(self.conv2(c12), scale_factor=2)), 1)
        # c11 = self.smooth1(c11)

        # c25 = torch.cat((c25, F.upsample_bilinear(self.conv6(c26), scale_factor=2)), 1)
        # c25 = self.smooth5(c25)
        # c24 = torch.cat((c24, F.upsample_bilinear(self.conv5(c25), scale_factor=2)), 1)
        # c24 = self.smooth4(c24)
        # c23 = torch.cat((c23, F.upsample_bilinear(self.conv4(c24), scale_factor=2)), 1)
        # c23 = self.smooth3(c23)
        # c22 = torch.cat((c22, F.upsample_bilinear(self.conv3(c23), scale_factor=2)), 1)
        # c22 = self.smooth2(c22)
        # c21 = torch.cat((c21, F.upsample_bilinear(self.conv2(c22), scale_factor=2)), 1)
        # c21 = self.smooth1(c21)

        c15 = c15 + F.upsample_bilinear(self.conv6(c16), scale_factor=2)
        c15 = self.smooth5(c15)
        c14 = c14 + F.upsample_bilinear(self.conv5(c15), scale_factor=2)
        c14 = self.smooth4(c14)
        c13 = c13 + F.upsample_bilinear(self.conv4(c14), scale_factor=2)
        c13 = self.smooth3(c13)
        c12 = c12 + F.upsample_bilinear(self.conv3(c13), scale_factor=2)
        c12 = self.smooth2(c12)
        c11 = c11 + F.upsample_bilinear(self.conv2(c12), scale_factor=2)
        c11 = self.smooth1(c11)

        c25 = c25 + F.upsample_bilinear(self.conv6(c26), scale_factor=2)
        c25 = self.smooth5(c25)
        c24 = c24 + F.upsample_bilinear(self.conv5(c25), scale_factor=2)
        c24 = self.smooth4(c24)
        c23 = c23 + F.upsample_bilinear(self.conv4(c24), scale_factor=2)
        c23 = self.smooth3(c23)
        c22 = c22 + F.upsample_bilinear(self.conv3(c23), scale_factor=2)
        c22 = self.smooth2(c22)
        c21 = c21 + F.upsample_bilinear(self.conv2(c22), scale_factor=2)
        c21 = self.smooth1(c21)
        return [c11, c21, c12, c22, c13, c23, c14, c24, c15, c25, c16, c26]

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = conv_norm(16, 32, kernel_size=4, stride=2, padding=1)
#         self.conv2 = conv_norm(32*2, 64*2, kernel_size=4, stride=2, padding=1)
#         self.conv3 = conv_norm(64*3, 96*3, kernel_size=4, stride=2, padding=1)
#         self.conv4 = conv_norm(96*4, 128*4, kernel_size=4, stride=2, padding=1)
#         self.conv5 = conv_norm(128*5, 196*5, kernel_size=4, stride=2, padding=1)
#         self.conv_final = nn.Conv2d(196*6, 1, kernel_size = 4, stride=1, padding=0)

#     def forward(self, x):
#         c11, c21, c12, c22, c13, c23, c14, c24, c15, c25, c16, c26 = x
#         c11 = self.conv1(c11)
#         c12 = torch.cat((c11, c12), 1)
#         c12 = self.conv2(c12)
#         c13 = torch.cat((c12, c13), 1)
#         c13 = self.conv3(c13)
#         c14 = torch.cat((c13, c14), 1)
#         c14 = self.conv4(c14)
#         c15 = torch.cat((c14, c15), 1)
#         c15 = self.conv5(c15)
#         c16 = torch.cat((c15, c16), 1)
#         result = self.conv_final(c16)
#         return result

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = conv_norm(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = conv_norm(32*2, 64*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = conv_norm(64*3, 96*3, kernel_size=4, stride=2, padding=1)
        self.conv4 = conv_norm(96*4, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        c11, c21, c12, c22, c13, c23, c14, c24, c15, c25, c16, c26 = x
        c11 = self.conv1(c11)
        c12 = torch.cat((c11, c12), 1)
        c12 = self.conv2(c12)
        c13 = torch.cat((c12, c13), 1)
        c13 = self.conv3(c13)
        c14 = torch.cat((c13, c14), 1)
        output = self.conv4(c14)

        return output

class Generator(nn.Module):
    def __init__(self, md = 4):
        super(Generator, self).__init__()
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=md*2+1, stride=1, padding=md, dilation_patch=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(int(od + dd[4]), 2, kernel_size=4, stride=2, padding=1)

        od = nd + 128 + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(int(od + dd[4]), 2, kernel_size=4, stride=2, padding=1)

        od = nd + 96 + 4
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(int(od + dd[4]), 2, kernel_size=4, stride=2, padding=1)

        od = nd + 64 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(int(od + dd[4]), 2, kernel_size=4, stride=2, padding=1)

        od = nd + 32 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, x):
        c11, c21, c12, c22, c13, c23, c14, c24, c15, c25, c16, c26 = x
        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
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
        corr5 = self.leakyRELU(corr5)
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
        corr4 = self.leakyRELU(corr4)
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
        corr3 = self.leakyRELU(corr3)

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
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        flow1 = F.upsample_bilinear(flow2, scale_factor=4)

        return flow1, [flow2, flow3, flow4, flow5, flow6]

class PWCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCNet,self).__init__()

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=md*2+1, stride=1, padding=md, dilation_patch=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask


    def forward(self,im1, im2):
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        corr6 = self.corr(c16, c26) 
        corr6 = self.leakyRELU(corr6)   


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        flow1 = F.upsample_bilinear(flow2, scale_factor=4)

        return flow1, [flow2,flow3,flow4,flow5,flow6]


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
