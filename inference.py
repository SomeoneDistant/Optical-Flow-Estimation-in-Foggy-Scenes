import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
from visdom import Visdom
import cv2
import numpy as np
import itertools
import os

import dataset
import model_new
import flow2img
import loss

from PIL import Image

if __name__ == '__main__':

    show_flow = flow2img.Flow()

    OpticalFlow = model_new.OpticalFlow()
    if torch.cuda.device_count() > 1:
        OpticalFlow = nn.DataParallel(OpticalFlow).cuda()
        print('Data-Parallel Complete.')
        BATCH_SIZE = 1 * torch.cuda.device_count()
        print('Paralleled Batch Size is %d. Number of GPU is %d'%(BATCH_SIZE,torch.cuda.device_count()))
    else:
        print('No Data-Parallel.')
        OpticalFlow = OpticalFlow.cuda()
        BATCH_SIZE = 1

    OpticalFlow.load_state_dict(torch.load('/home/yaoyuan/Desktop/viml11/HAZEFLOWNET/Hazeflownet_OpticalFlow.pth.tar'))
    OpticalFlow.eval()

    # DATASET = dataset.FoggyZurich(root='/home/yaoyuan/Dataset/Foggy_Zurich')
    # DATASET = dataset.VirtualKITTI(root='/home/yaoyuan/Desktop/viml11/Dataset/VirtualKITTI')
    DATASET = dataset.FlyingThings(root='/home/yaoyuan/fog_simulation/flyingthings')
    # DATASET = dataset.FlyingChairs(root='/home/yaoyuan/fog_simulation/FlyingChairs_release/data')
    # DATASET = dataset.MpiSintel(root='/home/yaoyuan/flownet2-pytorch-previous/MPI-Sintel-complete/training')

    DATALOADER = data.DataLoader(
        dataset=DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12
    )

    EPE_SUM = 0
    with torch.no_grad():
        for batch_idx, [img, label] in enumerate(DATALOADER):

            img_1 = Variable(img[0][:,:,0,:,:].cuda())
            img_2 = Variable(img[0][:,:,1,:,:].cuda())

            _, flow_1 = OpticalFlow(img_1, img_2)
            flow_1 = flow_1[0] * 20

            # flow_1, _ = OpticalFlow(img_1, img_2)
            # flow_1 = flow_1 * 20

            # # cv2 visualization
            # frame_1 = img[0].numpy()[0,:,0,:,:].astype('uint8')
            # frame_1 = frame_1.transpose(1,2,0)
            # frame_2 = img[0].numpy()[0,:,1,:,:].astype('uint8')
            # frame_2 = frame_2.transpose(1,2,0)
            # cv2.imshow('Frame 1', frame_1)
            # cv2.imshow('Frame 2', frame_2)
            # cv2.imshow('Estimation', show_flow._flowToColor(flow_1[0].data.cpu().numpy())) 
            # cv2.imshow('Ground Truth', show_flow._flowToColor(label[0][0].cpu().numpy()))
            # cv2.waitKey(delay = 1000)

            # target = label[0]
            target = nn.functional.interpolate(label[0], scale_factor=0.25, mode='bilinear')

            # for idx in range(flow_1.shape[0]):
            #     est_flow = Image.fromarray(show_flow._flowToColor(flow_1[idx].data.cpu().numpy()))
            #     est_flow.save('/home/yaoyuan/Desktop/viml11/HAZEFLOWNET/work/flo/sintel/%06d.png'%(batch_idx*BATCH_SIZE+idx))

            EPE_SUM += loss.realEPE(flow_1, target)
            EPE = EPE_SUM / (batch_idx + 1)
            print(
                '[Batch %d/%d] [EPE %f]' % (
                    batch_idx, len(DATALOADER), EPE
                    )
                )

