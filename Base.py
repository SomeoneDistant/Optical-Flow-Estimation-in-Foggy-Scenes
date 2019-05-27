import shutil
import datetime
import models
import flow_transform

from helper import *
from flowlib import *
from collections import OrderedDict
from skimage.transform import resize
from torchvision import transforms
from tensorboard_logger import configure
from torch.utils.data import DataLoader


class Base(object):
    def __init__(self, config):
        # start from basic parameters
        self.best_acc = 0

        self.config = config
        self.gpuid = config.gpuid
        self.dataset = config.dataset
        self.criterionMSE = nn.MSELoss().cuda(self.gpuid)
        self.criterionL1 = nn.L1Loss().cuda(self.gpuid)
        self.LR = config.learning_rate
        self.parallel = config.parallel
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.ckpt_dir = config.check_point_dir
        self.model_name = os.path.basename(os.getcwd()) + '_' + str(datetime.datetime.now())
        self.best_valid_acc = 0
        self.logs_dir = '.logs/'
        self.epoch_limit = config.epoch_limit
        self.use_tensorboard = True
        self.epoch = 1
        self.total_loss = 0
        self.outputdir = 'out/'
        self.valdir = 'val/'

        self.test_input_dir = config.test_input_dir
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.test_dir = config.test_dir
        self.train_sample_len = 0

        # init tensors
        self.image_in = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        self.input_list = []

        # declaration of outputs
        self.output_flow_list = []
        self.flow_pyr = []

        # == simple rain feature extractors ==
        self.net = models.get_pwcnet().cuda(self.gpuid)
        #self.net = models.pwc_dc_net('.ckpt/pwc_net_ckpt.pth.tar').cuda(self.gpuid)
        if self.parallel:
            self.net = torch.nn.DataParallel(self.net)
        self.net_optim = torch.optim.Adam(self.net.parameters(), lr=self.LR, betas=(0.9, 0.999))
        self.reset()

    # Setup TensorBoard
    def reset(self):
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

    def write_image_switch(self, path):
        im_in1, im_in2 = self.input_list[0], self.input_list[1]
        gtflow = tensor_to_flow(self.input_list[2])
        outflow = tensor_to_flow(self.output_flow_list[0])
        [_, _, h, w] = im_in1.size()
        flow_arr_est = resize(flow_to_image(outflow).astype(np.float32)/255.0, [h, w], mode='reflect').astype(np.float32)
        flow_arr_gt = resize(flow_to_image(gtflow).astype(np.float32)/255.0, [h, w], mode='reflect').astype(np.float32)
        img_painter = tensor_to_image(torch.cat([im_in1, im_in2], dim=3))
        flow_painter = np.concatenate([flow_arr_est, flow_arr_gt], axis=1)
        painter = np.concatenate([img_painter, flow_painter], axis=0)
        painter = np.clip(painter * 255, 0, 255)
        painter_image = Image.fromarray(painter.astype(np.uint8))
        painter_image.save(path)

    def load_data(self, mode, aug=False):
        if mode == 'train':
            if self.dataset == 'flyingchair':
                dataset = FlyingChairDataset(self.train_dir, 'train',
                                             transform=transforms.Compose([ToTensor()]),
                                             co_transform=flow_transform.Compose([
                                                 # flow_transform.RandomTranslation(0.03),
                                                 # flow_transform.RandomRotation(12, 5),
                                                 # flow_transform.RandomScale(0.9, 2),
                                                 # flow_transform.RandomCrop((320, 448), (360, 500)),
                                                 flow_transform.RandomAffineTransformation(0.9, 2.0, 0.05, 12, 5),
                                                 flow_transform.RandomConstraintCrop((320, 448), (360, 500)),
                                                 flow_transform.RandomVerticalFlip(),
                                                 flow_transform.RandomHorizontalFlip(),
                                                 flow_transform.ContrastAdjust(0.8, 1.4),
                                                 flow_transform.GammaAdjust(0.7, 1.5),
                                                 flow_transform.BrightnessAdjust(0, 0.2),
                                                 flow_transform.SaturationAdjust(0.5, 2),
                                                 flow_transform.HueAdjust(-0.2, 0.2)
                                             ])
                                             )
            elif self.dataset == 'flyingthings':
                dataset = FlyingChairDataset(self.train_dir, 'train',
                                             transform=transforms.Compose([ToTensor()]),
                                             co_transform=flow_transform.Compose([
                                                 flow_transform.RandomRotation(10),
                                                 flow_transform.RandomRotate(10, 5),
                                                 flow_transform.RandomCrop((384, 768)),
                                                 flow_transform.RandomVerticalFlip(),
                                                 flow_transform.RandomHorizontalFlip(),
                                                 flow_transform.RandomGamma(0.7, 1.5),
                                                 flow_transform.MultiplicativeColor(0.5, 2),
                                                 flow_transform.GaussianIllumination(0, 0.2)
                                             ])
                                             )
            shuff = True
        elif mode == 'val':
            dataset = FlyingChairDataset(self.val_dir, 'val',
                                        transform=transforms.Compose([ToTensor()]), 
                                        co_transform=flow_transform.Compose([
                                             flow_transform.RandomCrop((384, 448))])
                                        )
            shuff = False
        elif mode =='test':
            dataset = FlyingChairDataset(self.test_dir, 'test',
                                         transform=transforms.Compose([ToTensor()])
                                         # co_transform=flow_transform.Compose([
                                             # flow_transform.Scale((448, 1024))])
                                         )
            shuff = False
        else:
            dataset = None
            shuff = False
            print('Undefined mode', mode)
            exit()
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=shuff,
                                 drop_last=True,
                                 num_workers=self.batch_size)
        return data_loader

    def save_checkpoint(self, state, msg, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + msg + '.pth.tar'
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, msg, best=False, load_lr=True):
        print("[*] Loading model from {}{}.pth.tar".format(self.ckpt_dir, msg))
        filename = msg + '.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        if 'state_dict' in ckpt.keys():
            pretrained_weights = ckpt['state_dict']
        else:
            pretrained_weights = ckpt

        new_state_dict = OrderedDict()
        if self.parallel:   # expect module in the pretrained_weights
            for k, v, in pretrained_weights.items():
                if 'module' not in k:
                    name = 'module.'+k
                    new_state_dict[name] = v
                else:
                    new_state_dict = pretrained_weights
        else:
            for k, v in pretrained_weights.items():
                if 'module' in k:
                    name = k[7:]
                    new_state_dict[name] = v
                else:
                    new_state_dict = pretrained_weights

        self.net.load_state_dict(new_state_dict)
        if 'epoch' in ckpt.keys():
            self.epoch = ckpt['epoch']
        if 'best_valid_acc' in ckpt.keys():
            self.best_valid_acc = ckpt['best_valid_acc']

    def load_my_state_dict(self, state_dict):
        own_state = self.net.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)