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


def saveCKPT(dir, filename, model):
    if not os.path.exists(dir):
        os.mkdir(dir)
    ckpt_path = os.path.join(dir, filename)
    torch.save(model.state_dict(), ckpt_path)

if __name__ == '__main__':
    vis = Visdom(env='yaoyuan')

    OpticalFlow = model_new.OpticalFlow()

    if torch.cuda.device_count() > 1:
        OpticalFlow = nn.DataParallel(OpticalFlow).cuda()
        print('Data-Parallel Complete.')
        BATCH_SIZE = 8 * torch.cuda.device_count()
        print('Paralleled Batch Size is %d. Number of GPU is %d'%(BATCH_SIZE,torch.cuda.device_count()))
    else:
        print('No Data-Parallel.')
        OpticalFlow = OpticalFlow.cuda()
        BATCH_SIZE = 8

    EPOCH_SIZE = 90 # 450

    Optimizer = torch.optim.Adam(OpticalFlow.parameters(), lr=0.00001)
    # Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=[150, 225, 300, 375, 450], gamma=0.5)
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=[30, 45, 60, 75, 90], gamma=0.5)

    # DATASET = dataset.FoggyZurich(root='/home/yaoyuan/Dataset/Foggy_Zurich')
    # DATASET = dataset.VirtualKITTI(root='/home/yaoyuan/Dataset/VirtualKITTI')

    DATASET = dataset.FlyingThings(root='/home/yaoyuan/fog_simulation/flyingthings')
    # DATASET = dataset.FlyingChairs(root='/home/yaoyuan/fog_simulation/FlyingChairs_release/data')

    DATALOADER = data.DataLoader(
        dataset=DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=24
    )

    Initial = False
    if Initial:
        model_new.model_initial(OpticalFlow)
    else:
        OpticalFlow.load_state_dict(torch.load('/home/yaoyuan/Desktop/viml11/HAZEFLOWNET/Hazeflownet_OpticalFlow.pth.tar'))

    show_flow = flow2img.Flow()

    OpticalFlow.train()

    idx = 0
    vis_1 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Frame 1'))
    vis_2 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Frame 2'))
    vis_E1 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Estimation'))
    vis_G1 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Ground Truth'))
    vis_3 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Frame 1'))
    vis_4 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Frame 2'))
    vis_E2 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Estimation'))
    vis_G2 = vis.images(np.random.rand(4, 3, 256, 448), opts=dict(title='Ground Truth'))
    vis_L = vis.line(X=[idx], Y=[0], opts=dict(title='Train Loss'))

    for epoch_idx in range(EPOCH_SIZE):
        Scheduler.step()
        for batch_idx, [img, label] in enumerate(DATALOADER):

            frame_1 = img[0].numpy()[:,:,0,:,:].astype('uint8')
            frame_2 = img[0].numpy()[:,:,1,:,:].astype('uint8')

            # cv2 visualization
            # frame_1 = img[0].numpy()[0,:,0,:,:].astype('uint8')
            # frame_1 = frame_1.transpose(1,2,0)
            # frame_2 = img[0].numpy()[0,:,1,:,:].astype('uint8')
            # frame_2 = frame_2.transpose(1,2,0)
            # cv2.imshow('Frame 1', frame_1)
            # cv2.imshow('Frame 2', frame_2)
            # cv2.imshow('Estimation', show_flow._flowToColor(flow_1[0].data.cpu().numpy())) 
            # cv2.imshow('Ground Truth', show_flow._flowToColor(label[0][0].cpu().numpy()))
            # cv2.waitKey(delay = 1000)

            Optimizer.zero_grad()
            img_1 = Variable(img[0][:,:,0,:,:].cuda(), requires_grad=True)
            img_2 = Variable(img[0][:,:,1,:,:].cuda(), requires_grad=True)
            flow_1, output = OpticalFlow(img_1, img_2)
            Loss = loss.multiscaleEPE(output, label[0]/20)
            Loss.backward()
            Optimizer.step()
            EPE = loss.realEPE(flow_1*20, label[0])
            print(
                '[Epoch %d/%d] [Batch %d/%d] [loss %f] [EPE %f]' % (
                    epoch_idx, EPOCH_SIZE, batch_idx, len(DATALOADER), Loss.item(), EPE
                    )
                )

            if batch_idx == 0:
                flow_estimation = []
                flow_groundtruth = []
                for flow_idx in range(label[0].shape[0]):
                    flow_estimation.append(show_flow._flowToColor(flow_1[flow_idx].data.cpu().numpy()).transpose(2,0,1))
                    flow_groundtruth.append(show_flow._flowToColor(label[0][flow_idx].cpu().numpy()).transpose(2,0,1))
                vis.line(X=[epoch_idx], Y=[Loss.item()], win=vis_L, update='append')

                vis.images(frame_1[0:4,:,:,:], win=vis_1)
                vis.images(frame_2[0:4,:,:,:], win=vis_2)
                vis.images(flow_estimation[0:4], win=vis_E1)
                vis.images(flow_groundtruth[0:4], win=vis_G1)

                vis.images(frame_1[4:8,:,:,:], win=vis_3)
                vis.images(frame_2[4:8,:,:,:], win=vis_4)
                vis.images(flow_estimation[4:8], win=vis_E2)
                vis.images(flow_groundtruth[4:8], win=vis_G2)

        saveCKPT('/home/yaoyuan/Desktop/viml11/HAZEFLOWNET', 'Hazeflownet_OpticalFlow.pth.tar', OpticalFlow)

    flow_estimation = []
    flow_groundtruth = []
    for flow_idx in range(label[0].shape[0]):
        flow_estimation.append(show_flow._flowToColor(flow_1[flow_idx].data.cpu().numpy()).transpose(2,0,1))
        flow_groundtruth.append(show_flow._flowToColor(label[0][flow_idx].cpu().numpy()).transpose(2,0,1))
    vis.line(X=[epoch_idx+1], Y=[Loss.item()], win=vis_L, update='append')

    vis.images(frame_1[0:4,:,:,:], win=vis_1)
    vis.images(frame_2[0:4,:,:,:], win=vis_2)
    vis.images(flow_estimation[0:4], win=vis_E1)
    vis.images(flow_groundtruth[0:4], win=vis_G1)

    vis.images(frame_1[4:8,:,:,:], win=vis_3)
    vis.images(frame_2[4:8,:,:,:], win=vis_4)
    vis.images(flow_estimation[4:8], win=vis_E2)
    vis.images(flow_groundtruth[4:8], win=vis_G2)


