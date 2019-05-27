from __future__ import division
import cv2
import torch
import random
import numpy as np
import numbers
import types
from PIL import Image
import torchvision.transforms.functional as F
import scipy.ndimage as ndimage

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    co_transforms.Compose([
    co_transforms.CenterCrop(10),
    co_transforms.ToTensor(),
    ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs[0].shape
        h2, w2, _ = inputs[1].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        x2 = int(round((w2 - tw) / 2.))
        y2 = int(round((h2 - th) / 2.))

        inputs[0] = inputs[0][y1: y1 + th, x1: x1 + tw]
        inputs[1] = inputs[1][y2: y2 + th, x2: x2 + tw]
        target = target[y1: y1 + th, x1: x1 + tw]
        return inputs,target


class RandomAffineTransformation(object):
    def __init__(self, scale_lb, scale_ub, translation_range, init_angle_range, rotation_angle_range):
        self.scale_lb = scale_lb
        self.scale_ub = scale_ub
        #self.scale_factor = scale_factor
        self.translation_range = translation_range
        self.init_angle_range = init_angle_range
        self.rotation_angle_range = rotation_angle_range
        self.curr_w = None
        self.curr_h = None
        self.rotation_angle = None
        self.init_angle = None

    def __call__(self, inputs, target):
        """

        :param inputs: input image pairs
        :param target: target optical flow field
        :return: transformed inputs and target
        """
        h, w, c = inputs[0].shape
        # Translation
        trans_h_range = round(h*self.translation_range)
        trans_w_range = round(w*self.translation_range)
        th = random.randint(-trans_h_range, trans_h_range)
        tw = random.randint(-trans_w_range, trans_w_range)
        # th = self.translation_range[0]
        # tw = self.translation_range[1]
        if tw != 0 or th !=0:
            x1, x2, x3, x4 = max(0, tw), min(w+tw, w), max(0, -tw), min(w-tw, w)
            y1, y2, y3, y4 = max(0, th), min(h+th, h), max(0, -th), min(h-th, h)

            inputs[0] = inputs[0][y1:y2, x1:x2]
            inputs[1] = inputs[1][y3:y4, x3:x4]
            target = target[y1:y2, x1:x2]
            target[:,:,0] += tw
            target[:,:,1] += th

        assert(inputs[0].shape == inputs[1].shape)
        assert(inputs[0].shape[0] == target.shape[0])
        assert(inputs[0].shape[1] == target.shape[1])

        # Rotation : rotation angles determines the lower bound of Random Scale
        self.curr_h, self.curr_w, c = inputs[0].shape
        self.init_angle = random.uniform(-self.init_angle_range, self.init_angle_range)
        self.rotation_angle = random.uniform(-self.rotation_angle_range, self.rotation_angle_range)
        self.init_angle_rad = self.init_angle * np.pi/180
        self.rotation_angle_rad = self.rotation_angle * np.pi/180
        delta_flow_fields = np.fromfunction(self.compute_flow_field, target.shape)
        target += delta_flow_fields

        # reset first image and flow to new position
        M1 = cv2.getRotationMatrix2D((self.curr_w/2, self.curr_h/2), self.init_angle, 1)
        inputs[0] = cv2.warpAffine(inputs[0], M1, (self.curr_w, self.curr_h))
        target = cv2.warpAffine(target, M1, (self.curr_w, self.curr_h))

        # make rotation
        M_rot = cv2.getRotationMatrix2D((self.curr_w/2, self.curr_h/2), self.rotation_angle+self.init_angle, 1)
        inputs[1] = cv2.warpAffine(inputs[1], M_rot, (self.curr_w, self.curr_h))

        # Should consider the rotation of new initial position
        target_ = np.copy(target)
        target[:,:,0] = np.cos(self.init_angle_rad) * target_[:,:,0] + np.sin(self.init_angle_rad)*target_[:,:,1]
        target[:,:,1] = -np.sin(self.init_angle_rad) * target_[:,:,0] + np.cos(self.init_angle_rad) * target_[:,:,1]

        # Scale
        # find the range of Random zoom in operation
        # in order to cut the new rectangle in the boundary of the rotated image
        # we need to make sure the four corners of new rectangle are in the rotated image 4 bounds
        # at this moment we already know the rotating angles
        lb1 = self.get_lower_bound(self.init_angle_rad)
        lb2 = self.get_lower_bound(self.init_angle_rad+self.rotation_angle_rad)
        larger_lower_bound = max(lb1, lb2)

        self.scale_lb = max(self.scale_lb, larger_lower_bound)
        scale_value = np.random.uniform(self.scale_lb, self.scale_ub)
        inputs[0] = cv2.resize(inputs[0], None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_LINEAR)
        inputs[1] = cv2.resize(inputs[1], None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_LINEAR)

        target = cv2.resize(target, None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_LINEAR)
        target[0] = target[0] * scale_value
        target[1] = target[1] * scale_value

        return inputs, target

    def get_lower_bound(self, angle1):
        """
        have a (-250, -180) -> (250, 180) rectangle
        :param angle1: rotation angle of first image
        :param angle2: rotation angle of second image
        :return:
        """
        z_min = 0

        if angle1 > 0:
            # Point 1
            l_new = (180 + 250 / np.tan(angle1)) * np.sin(angle1)
            l_old = self.curr_w / 2
            lower_bound = abs(l_new / l_old)
            if lower_bound > z_min:
                z_min = lower_bound
            # Point 2
            s_new = (180 + 250 * np.tan(angle1)) * np.cos(angle1)
            s_old = self.curr_h / 2
            lower_bound = abs(s_new / s_old)
            if lower_bound > z_min:
                z_min = lower_bound

        else:
            # Point 1
            s_new = (180+250*np.tan(-angle1))* np.cos(-angle1)
            s_old = self.curr_h / 2
            lower_bound = abs(s_new/s_old)
            if lower_bound > z_min:
                z_min = lower_bound
            # Point2
            l_new = (180 + 250 / np.tan(-angle1)) * np.sin(-angle1)
            l_old = self.curr_w / 2
            lower_bound = abs(l_new / l_old)
            if lower_bound > z_min:
                z_min = lower_bound

        return z_min

    def compute_flow_field(self, i, j, k):
        return -k * (j - self.curr_w / 2) * self.rotation_angle_rad + (1 - k) * (i - self.curr_h / 2) * self.rotation_angle_rad


class RandomScale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, scale_factor):
        #self.lv = lv
        #self.hv = hv
        self.scale_factor = scale_factor

    def __call__(self, inputs, target):
        #scale_value = np.random.uniform(self.lv, self.hv)
        scale_value = self.scale_factor
        inputs[0] = cv2.resize(inputs[0], None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_LINEAR)
        inputs[1] = cv2.resize(inputs[1], None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_LINEAR)

        target = cv2.resize(target, None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_LINEAR)
        target[0] = target[0] * scale_value
        target[1] = target[1] * scale_value
        return inputs, target


class RandomConstraintCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, constraint_region):
        """

        :param size: crop image to size = [h,w]
        :param constraint_region: select crops from this region [ch, cw]
        """
        self.ch = size[0]
        self.cw = size[1]
        self.bh = constraint_region[0]
        self.bw = constraint_region[1]

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        x1 = random.randint(round((w-self.bw)/2), round((w+self.bw)/2 - self.cw))
        y1 = random.randint(round((h-self.bh)/2), round((h+self.bh)/2 - self.ch))
        inputs[0] = inputs[0][y1:y1+self.ch, x1:x1+self.cw]
        inputs[1] = inputs[1][y1:y1+self.ch, x1:x1+self.cw]
        return inputs, target[y1:y1+self.ch, x1:x1+self.cw]


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs,target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0][y1: y1 + th,x1: x1 + tw]
        inputs[1] = inputs[1][y1: y1 + th,x1: x1 + tw]
        return inputs, target[y1: y1 + th,x1: x1 + tw]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.fliplr(inputs[0]))
            inputs[1] = np.copy(np.fliplr(inputs[1]))
            target = np.copy(np.fliplr(target))
            target[:,:,0] *= -1
        return inputs,target


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.flipud(inputs[0]))
            inputs[1] = np.copy(np.flipud(inputs[1]))
            target = np.copy(np.flipud(target))
            target[:,:,1] *= -1
        return inputs,target


class RandomTranslation(object):
    def __init__(self, perc):
        self.perc = perc

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        # h_limit = round(h * self.perc)
        # w_limit = round(w * self.perc)
        # th = random.randint(-h_limit, h_limit)
        # tw = random.randint(-w_limit, w_limit)
        th = self.perc[0]
        tw = self.perc[1]
        if tw == 0 and th == 0:
            return inputs, target
        # compute do real translate
        M = np.float32([[1, 0, tw], [0, 1, th]])
        inputs[1] = cv2.warpAffine(inputs[1], M, (w, h))
        target[:,:,0] += tw
        target[:,:,1] += th
        return inputs, target


class RandomRotation(object):
    def __init__(self, init_angle_limit, rotate_angle_limit):
        self.rotate_angle_limit = rotate_angle_limit
        self.init_angle_limit = init_angle_limit

    def __call__(self, inputs, target):
        #init_angle = random.uniform(-self.init_angle_limit, self.init_angle_limit)
        #rotate_angle = random.uniform(-self.rotate_angle_limit, self.rotate_angle_limit)
        init_angle = self.init_angle_limit
        rotate_angle = self.rotate_angle_limit
        init_angle_rad = init_angle * np.pi / 180
        h, w, c = inputs[0].shape

        def compute_flow_field(i,j,k):
            return -k * (j - w / 2) * (rotate_angle * np.pi / 180) + (1 - k) * (i - h / 2) * (rotate_angle * np.pi / 180)
        delta_flow_fields = np.fromfunction(compute_flow_field, target.shape)
        target += delta_flow_fields

        # reset first image new position
        M_init = cv2.getRotationMatrix2D((w/2, h/2), init_angle, 1)
        inputs[0] = cv2.warpAffine(inputs[0], M_init, (w, h))
        # make rotation
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angle+init_angle, 1)
        inputs[1] = cv2.warpAffine(inputs[1], M, (w, h))
        target = cv2.warpAffine(target, M_init, (w, h))

        target_ = np.copy(target)
        target[:,:,0] = np.cos(init_angle_rad)*target_[:,:,0] + np.sin(init_angle_rad)*target_[:,:,1]
        target[:,:,1] = -np.sin(init_angle_rad)*target_[:,:,0] + np.cos(init_angle_rad)*target_[:,:,1]

        return inputs, target


class RandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, inputs, target):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        inputs[0] *= (1 + random_std)
        inputs[0] += random_mean

        inputs[1] *= (1 + random_std)
        inputs[1] += random_mean

        inputs[0] = inputs[0][:,:,random_order]
        inputs[1] = inputs[1][:,:,random_order]

        return inputs, target


class GaussianIllumination(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, inputs, target):
        additive = np.random.normal(self.mu, self.sigma, 1)
        inputs[0] = np.clip(inputs[0] + additive, 0, 1).astype(np.float32)
        inputs[1] = np.clip(inputs[1] + additive, 0, 1).astype(np.float32)
        return inputs, target


class ContrastAdjust(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, inputs, target):
        contrast_factor = np.random.uniform(self.low, self.high)
        if not isinstance(inputs[0], Image.Image):
            a1 = np.clip(inputs[0]*255, 0, 255).astype(np.uint8)
            a2 = np.clip(inputs[1]*255, 0, 255).astype(np.uint8)
            inputs[0] = Image.fromarray(a1)
            inputs[1] = Image.fromarray(a2)
        inputs[0] = F.adjust_contrast(inputs[0], contrast_factor)
        inputs[1] = F.adjust_contrast(inputs[1], contrast_factor)
        return inputs, target


class GammaAdjust(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, inputs, target):
        gamma = np.random.uniform(self.low, self.high)
        inputs[0] = F.adjust_gamma(inputs[0], gamma)
        inputs[1] = F.adjust_gamma(inputs[1], gamma)
        return inputs, target


class BrightnessAdjust(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, inputs, target):
        brightness = np.random.normal(self.mu, self.sigma)
        inputs[0] = F.adjust_brightness(inputs[0], 1+brightness)
        inputs[1] = F.adjust_brightness(inputs[1], 1+brightness)
        return inputs, target


class SaturationAdjust(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, inputs, target):
        saturation = np.random.uniform(self.low, self.high)
        inputs[0] = F.adjust_saturation(inputs[0], saturation)
        inputs[1] = F.adjust_saturation(inputs[1], saturation)
        return inputs, target


class HueAdjust(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, inputs, target):
        hue = np.random.uniform(self.low, self.high)
        inputs[0] = F.adjust_hue(inputs[0], hue)
        inputs[1] = F.adjust_hue(inputs[1], hue)
        return inputs, target


# ==========================================
class RandomGamma(object):
    def __init__(self, lb=0.7, hb=1.5):
        self.lb = lb
        self.hb = hb

    def __call__(self, inputs, target):
        gamma = np.random.uniform(self.lb, self.hb, 1)
        inputs[0] = np.power(np.clip(inputs[0],0, 1), gamma).astype(np.float32)
        inputs[1] = np.power(np.clip(inputs[1],0, 1), gamma).astype(np.float32)
        return inputs, target


class MultiplicativeColor(object):
    def __init__(self, lb, hb):
        self.lb = lb
        self.hb = hb

    def __call__(self, inputs, target):
        kernel = np.random.uniform(self.lb, self.hb, 3).reshape((1, 1, 3))
        inputs[0] = np.clip(inputs[0] * kernel, 0, 1).astype(np.float32)
        inputs[1] = np.clip(inputs[1] * kernel, 0, 1).astype(np.float32)
        return inputs, target


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs,target):
        #applied_angle = random.uniform(-self.angle,self.angle)
        # diff = random.uniform(-self.diff_angle,self.diff_angle)
        applied_angle = self.angle
        diff = self.diff_angle
        angle1 = applied_angle - diff/2
        angle2 = applied_angle + diff/2
        angle1_rad = angle1*np.pi/180

        h, w, _ = target.shape

        def rotate_flow(i,j,k):
            return -k*(j-w/2)*(diff*np.pi/180) + (1-k)*(i-h/2)*(diff*np.pi/180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape, order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=self.reshape, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        target_ = np.copy(target)
        target[:,:,0] = np.cos(angle1_rad)*target_[:,:,0] + np.sin(angle1_rad)*target_[:,:,1]
        target[:,:,1] = -np.sin(angle1_rad)*target_[:,:,0] + np.cos(angle1_rad)*target_[:,:,1]
        return inputs,target


class RandomTranslate(object):
    def __init__(self, ty, tx):
        # if isinstance(translation, numbers.Number):
        #     self.translation = (int(translation), int(translation))
        # else:
        #     self.translation = translation
        self.tx = tx
        self.ty = ty

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        # th, tw = self.translation
        # tw = random.randint(-tw, tw)
        # th = random.randint(-th, th)
        th = self.ty
        tw = self.tx
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th

        return inputs, target


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.h = size[0]
        self.w = size[1]
        self.order = order

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape

        #hratio = self.h/h
        #wratio = self.w/w
        hratio = self.h
        wratio = self.w

        inputs[0] = ndimage.interpolation.zoom(inputs[0], [hratio, wratio, 1], order=self.order)
        inputs[1] = ndimage.interpolation.zoom(inputs[1], [hratio, wratio, 1], order=self.order)

        target = ndimage.interpolation.zoom(target, [hratio, wratio, 1], order=self.order)
        target[0] = target[0] * wratio
        target[1] = target[1] * hratio
        return inputs, target


# ===========================================