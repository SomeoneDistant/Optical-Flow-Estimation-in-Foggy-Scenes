import torch
import torch.utils.data as data
from time import time

import os, math, random, re
from os.path import *
import numpy as np

from glob import glob

from scipy import ndimage
import cv2
from cv2 import imread

import flow_transform

# __all__ = ['VirtualKITTI']

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)
    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

# class Scale(object):
#     def __init__(self, image_size, crop_size):
#         self.th, self.tw = crop_size
#         self.h, self.w = image_size
#         # if (self.h * 0.5 < self.th) or (self.w * 0.5 < self.tw):
#         #     self.ratio = max(self.th / self.h, self.tw / self.w)
#         # else:
#         self.ratio = np.random.uniform(max(self.th / self.h, self.tw / self.w)+0.1, 2)

#     def __call__(self, img, choice):
#         chanel1 = ndimage.interpolation.zoom(img[:,:,0], self.ratio, order=2)
#         chanel2 = ndimage.interpolation.zoom(img[:,:,1], self.ratio, order=2)
#         if choice:
#             img = np.dstack([chanel1, chanel2])
#             img *= self.ratio
#         else:
#             chanel3 = ndimage.interpolation.zoom(img[:,:,2], self.ratio, order=2)
#             img = np.dstack([chanel1, chanel2, chanel3])
#         return img

# class RandomHorizontalFlip(object):
#     def __init__(self):
#         self.rand = random.randint(0,1)
#     def __call__(self, img, choice):
#         if self.rand:
#             img = np.copy(np.fliplr(img))
#             if choice:
#                 img[:,:,0] = -1 * img[:,:,0]
#         return img

# class RandomVerticalFlip(object):
#     def __init__(self):
#         self.rand = random.randint(0,1)
#     def __call__(self, img, choice):
#         if self.rand:
#             img = np.copy(np.flipud(img))
#             if choice:
#                 img[:,:,1] = -1 * img[:,:,1]
#         return img

# class RandomRotate(object):
#     def __init__(self):
#         self.angle = random.uniform(-180, 180)
#     def __call__(self, img, choice):
#         angle_rad = self.angle*np.pi/180
#         img = ndimage.interpolation.rotate(img, self.angle, reshape=False, order=2)
#         if choice:
#             img_ = np.copy(img)
#             img[:,:,0] = np.cos(angle_rad)*img_[:,:,0] + np.sin(angle_rad)*img_[:,:,1]
#             img[:,:,1] = -np.sin(angle_rad)*img_[:,:,0] + np.cos(angle_rad)*img_[:,:,1]
#         return img

# class RandomTranslate(object):
#     def __init__(self):
#         self.translation=[50,100]
#     def __call__(self, inputs, target):
#         h, w, _ = inputs[0].shape
#         th, tw = self.translation
#         tw = random.randint(-tw, tw)
#         th = random.randint(-th, th)
#         if tw == 0 and th == 0:
#             return inputs, target
#         x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
#         y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)
#         inputs[0] = inputs[0][y1:y2,x1:x2]
#         inputs[1] = inputs[1][y3:y4,x3:x4]
#         target = target[y1:y2,x1:x2]
#         target[:,:,0] += tw
#         target[:,:,1] += th
#         return inputs, target

class FlyingChairs(data.Dataset):
    def __init__(self, is_augment=False, root = '/path/to/FlyingChairs_release/data'):
        self.augmentation = flow_transform.Compose([
            flow_transform.RandomAffineTransformation(0.9, 2.0, 0.03, 12, 5),
            flow_transform.RandomConstraintCrop((256, 448), (300, 500)),
            flow_transform.RandomVerticalFlip(),
            flow_transform.RandomHorizontalFlip(),
            # flow_transform.ContrastAdjust(0.8, 1.4),
            # flow_transform.GammaAdjust(0.7, 1.5),
            # flow_transform.BrightnessAdjust(0, 0.2),
            # flow_transform.SaturationAdjust(0.5, 2),
            # flow_transform.HueAdjust(-0.2, 0.2)
            ])
        self.is_augment = is_augment
        # self.crop_size = [384, 448]

        images = sorted(glob(join(root, '*.ppm')))

        self.flow_list = sorted(glob(join(root, '*.flo')))

        assert (len(images)//2 == len(self.flow_list))

        self.image_list = []

        for i in range(len(self.flow_list)):
            im1 = images[i*2]
            im2 = images[i*2 + 1]

            self.image_list += [ [ im1, im2 ] ]

        assert len(self.image_list) == len(self.flow_list)

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

    def __getitem__(self, index):
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        # add fog
        # atmosphere = np.exp(-np.random.uniform(1,3))
        # scatter = np.random.uniform(50,150)
        # img1 = img1 * atmosphere + scatter * (1 - atmosphere)
        # img2 = img2 * atmosphere + scatter * (1 - atmosphere)

        flow = readFlow(self.flow_list[index])

        images = [img1, img2]

        if self.is_augment:
            images, flow = self.augmentation(images, flow)
            images[0] = np.array(images[0])
            images[1] = np.array(images[1])
        else:
            image_size = np.array(images[0].shape[:2])
            crop_size = (image_size // 64) * 64
            cropper = StaticCenterCrop(image_size, crop_size)
            images = list(map(cropper, images))
            flow = cropper(flow)

        assert (images[0].shape[:2] == images[1].shape[:2])

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size

class FlyingThings(data.Dataset):
    def __init__(self, is_augment=False, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass'):
        self.augmentation = flow_transform.Compose([
            flow_transform.RandomAffineTransformation(0.9, 1.5, 0.03, 12, 5),
            flow_transform.RandomConstraintCrop((256, 448), (300, 500)),
            flow_transform.RandomVerticalFlip(),
            flow_transform.RandomHorizontalFlip()
            # flow_transform.ContrastAdjust(0.8, 1.4),
            # flow_transform.GammaAdjust(0.7, 1.5),
            # flow_transform.BrightnessAdjust(0, 0.2),
            # flow_transform.SaturationAdjust(0.5, 2),
            # flow_transform.HueAdjust(-0.2, 0.2)
            ])
        self.is_augment = is_augment

        image_dirs = sorted(glob(join(root, dstype, 'TEST/*/*')))
        image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

        flow_dirs = sorted(glob(join(root, 'optical_flow/TEST/*/*')))
        flow_dirs = sorted([join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

        depth_dirs = sorted(glob(join(root, 'disparity/TEST/*/*')))
        depth_dirs = sorted([join(f, 'left') for f in depth_dirs] + [join(f, 'right') for f in depth_dirs])

        self.root_len = len(root) + len(dstype) + 2

        assert len(image_dirs) == len(flow_dirs) == len(depth_dirs)

        self.image_list = []
        self.flow_list = []
        self.depth_list = []
        self.ex_list = []

        ex_dirs = open(join(root, 'all_unused_files.txt'), 'r').readlines()
        for ex_num in range(len(ex_dirs)):
            self.ex_list += [ex_dirs[ex_num][:-1]]

        for idir, fdir, ddir in zip(image_dirs, flow_dirs, depth_dirs):
            images = sorted(glob(join(idir, '*.png')))
            flows = sorted(glob(join(fdir, '*.pfm')))
            depths = sorted(glob(join(ddir, '*.pfm')))
            for i in range(len(flows)-1):

                if images[i][self.root_len:] in self.ex_list or images[i+1][self.root_len:] in self.ex_list:
                    print('Exclusion Detected\n')
                    continue

                self.image_list += [[images[i], images[i+1]]]
                self.depth_list += [[depths[i], depths[i+1]]]
                self.flow_list += [flows[i]]

        assert len(self.image_list) == len(self.flow_list) == len(self.depth_list)

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

    def __getitem__(self, index):

        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        # dept1, _ = readPFM(self.depth_list[index][0])
        # dept1 = dept1[:,:,np.newaxis]
        # dept2, _ = readPFM(self.depth_list[index][1])
        # dept2 = dept2[:,:,np.newaxis]

        # dept1 = 1 / (dept1 + 0.001)
        # dept2 = 1 / (dept2 + 0.001)

        # add fog
        # b = np.random.uniform(1, 3)
        # scatter = np.random.uniform(50,150)

        # atmosphere1 = np.exp(-b * dept1)
        # atmosphere2 = np.exp(-b * dept2)
        # img1 = np.clip(img1 * atmosphere1 + scatter * (1 - atmosphere1), 0, 255)
        # img2 = np.clip(img2 * atmosphere2 + scatter * (1 - atmosphere2), 0, 255)

        flow, _ = readPFM(self.flow_list[index])
        flow = flow[:,:,0:2]

        images = [img1, img2]

        if self.is_augment:
            images, flow = self.augmentation(images, flow)
            images[0] = np.array(images[0])
            images[1] = np.array(images[1])
        else:
            image_size = np.array(images[0].shape[:2])
            crop_size = (image_size // 64) * 64
            cropper = StaticCenterCrop(image_size, crop_size)
            images = list(map(cropper, images))
            flow = cropper(flow)

        assert (images[0].shape[:2] == images[1].shape[:2])

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size

class VirtualKITTI(data.Dataset):
    def __init__(self, root = ''):

        flow_root = join(root, 'vkitti_1.3.1_flowgt')
        image_root = join(root, 'vkitti_1.3.1_rgb')

        file_list = sorted(glob(join(flow_root, '*/*.png')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            fbase = file[-14:]
            fnum = int(file[-9:-4])
            img1 = join(image_root, file[-14:])
            img2 = join(image_root, file[-14:-9]+"%05d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = read_vkitti_png_flow(self.flow_list[index])
        images = [img1, img2]

        assert (images[0].shape[:2] == images[1].shape[:2])

        image_size = np.array(images[0].shape[:2])
        crop_size = (image_size // 64) * 64
        cropper = StaticCenterCrop(image_size, crop_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size

class FoggyZurich(data.Dataset):
    def __init__(self, is_cropped = True, root = '', replicates = 1):
        self.is_cropped = is_cropped
        self.crop_size = [1024, 1024]
        self.replicates = replicates

        file_list = sorted(glob(join(root, '*/*.png')))

        self.image_list = []

        for file in file_list:
            fnum = int(file[-10:-4])
            img1 = file
            img2 = file[:-10]+"%06d"%(fnum+1) + '.png'

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]

        self.size = len(self.image_list)
        self.frame_size = read_gen(self.image_list[0][0]).shape

    def __getitem__(self, index):

        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        images = [img1, img2]

        assert (images[0].shape[:2] == images[1].shape[:2])

        image_size = images[0].shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
            images = list(map(cropper, images))

        images = np.array(images).transpose(3,0,1,2)
        images = torch.from_numpy(images.astype(np.float32))

        return [images]

    def __len__(self):
        return self.size * self.replicates

class MpiSintel(data.Dataset):
    def __init__(self, root = '', dstype = 'clean'):

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        # num = 0
        # for file in file_list:
        #     if 'test' in file:
        #         continue

        #     fbase = file[len(flow_root)+1:-14]
        #     fbase_num = int(file[-8:-4])
        #     if fbase_num == 1:
        #         num = num+1

        #     img1 = join(image_root, fbase + "%06d"%(num) + '.png')
        #     img2 = join(image_root, fbase + "%06d"%(num+1) + '.png')

        #     num = num + 1

        #     if not isfile(img1) or not isfile(img2) or not isfile(file):
        #         continue

        #     self.image_list += [[img1, img2]]
        #     self.flow_list += [file]

        self.size = len(self.image_list)

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = readFlow(self.flow_list[index])

        images = [img1, img2]

        image_size = np.array(images[0].shape[:2])
        crop_size = (image_size // 64) * 64
        cropper = StaticCenterCrop(image_size, crop_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size

def read_vkitti_png_flow(flow_fn):
    bgr = imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag: == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0   # or another value (e.g., np.nan)
    return out_flow

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    # elif ext == '.bin' or ext == '.raw':
    #     return np.load(file_name)
    # elif ext == '.flo':
    #     return flow_utils.readFlow(file_name).astype(np.float32)
    return []

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
