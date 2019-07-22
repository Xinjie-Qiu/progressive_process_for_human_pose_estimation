import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import scipy.io
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as transforms_F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json
from pycocotools.coco import COCO
from os import path
import math
import torch.nn.functional as F
from scipy import ndimage
from numpy import matlib
from torch.optim import lr_scheduler
from apex import amp

from torch.nn.modules import loss
from skimage.feature import peak_local_max
from tensorboardX import SummaryWriter
import random
from torchstat import stat

matplotlib.use('TkAgg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nModules = 2
nFeats = 256
nStack = 3
nKeypoint_COCO = 17
nSkeleton_COCO = 19
nKeypoint_MPII = 16
nSkeleton_MPII = 15
nOutChannels_0 = 2
nOutChannels_1 = nSkeleton_MPII + 1
nOutChannels_2 = nKeypoint_MPII + 1
epochs = 300
batch_size = 30
keypoints = 17
skeleton = 20
inputsize = 256
learning_rate = 1e-4

threshold = 1

mode = 'test'
load_model_name = 'params_1_merge_all_fine_tune'
save_model_name = 'params_1_merge_all_fine_tune'
# load_mask_name = 'params_1_mask'
# save_mask_name = 'params_1_mask'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
val_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_val2017.json'
train_image_dir_coco = '/data/COCO/COCO2017/train2017/'
val_image_dir_coco = '/data/COCO/COCO2017/val2017'

loss_img = save_model_name[:-4] + 'loss.png'
accuracy_img = save_model_name[:-4] + 'accuracy.png'

rootdir = '/data/lsp_dataset/images/'
retrain = False
train_mask = False
usemask = False
write = True
fine_tune = False
dataset = 'mpii'

sks = [[0, 1],
       [1, 2],
       [2, 6],
       [6, 3],
       [3, 4],
       [4, 5],
       [6, 7],
       [7, 8],
       [8, 9],
       [10, 11],
       [11, 12],
       [12, 8],
       [8, 13],
       [13, 14],
       [14, 15]
       ]


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, segment, keypoints = sample['image'], sample['segment'], sample['keypoints']

        w, h = image.size[:2]

        new_w, new_h = self.output_size, self.output_size

        new_w, new_h = int(new_w), int(new_h)

        img = image.resize([new_h, new_w])

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        for i in range(len(segment)):
            segment[i][0::2] = np.multiply(segment[i][0::2], new_w / w / 4)
            segment[i][1::2] = np.multiply(segment[i][1::2], new_h / h / 4)
            keypoints[i][0::3] = np.multiply(keypoints[i][0::3], new_w / w / 4)
            keypoints[i][1::3] = np.multiply(keypoints[i][1::3], new_h / h / 4)

        return {'image': img, 'segment': segment, 'keypoints': keypoints}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        image, segment, keypoints = sample['image'], sample['segment'], sample['keypoints']
        if random.random() < self.p:
            w, h = image.size[:2]
            image = transforms_F.hflip(image)
            for i in range(len(segment)):
                segment[i][0::2] = np.abs(np.subtract(segment[i][0::2], w / 4))
                # segment[i][1::2] = np.abs(np.subtract(segment[i][1::2], h))
                keypoints[i][0::3] = np.abs(np.subtract(keypoints[i][0::3], w / 4))
                # keypoints[i][1::3] = np.abs(np.subtract(keypoints[i][0::3], w))
        return {'image': image, 'segment': segment, 'keypoints': keypoints}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segment, keypoints = sample['image'], sample['segment'], sample['keypoints']

        w, h = image.size[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = Image.fromarray(np.array(image)[top: top + new_h, left: left + new_w])

        for i in range(len(segment)):
            segment[i][0::2] = np.maximum(np.subtract(segment[i][0::2], left / 4), 0)
            segment[i][1::2] = np.maximum(np.subtract(segment[i][1::2], top / 4), 0)
            keypoints[i][0::3] = np.maximum(np.subtract(keypoints[i][0::3], left / 4), 0)
            keypoints[i][1::3] = np.maximum(np.subtract(keypoints[i][1::3], top / 4), 0)

        return {'image': img, 'segment': segment, 'keypoints': keypoints}


class myImageDataset_COCO(data.Dataset):
    def __init__(self, anno, image_dir, transform=None):
        'Initialization'
        self.anno = COCO(anno)
        self.image_dir = image_dir
        self.lists = self.anno.getImgIds(catIds=self.anno.getCatIds())
        self.transform = transform

    def __len__(self):
        return len(self.lists)
        # return 1000

    def __getitem__(self, index):
        list = self.lists[index]
        sample = {}
        image_name = self.anno.loadImgs(list)[0]['file_name']
        image_path = path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        w, h = image.size
        sample['image'] = image
        # plt.imshow(image)
        # plt.show()
        label_id = self.anno.getAnnIds(list)
        labels = self.anno.loadAnns(label_id)

        segment_array = []
        keypoints_array = []
        draw = ImageDraw.Draw(image)
        for label in labels:
            try:
                segment = label['segmentation'][0]
                segment_array.append(segment)
                # seg_x = segment[0::2]
                # seg_y = segment[1::2]
                # draw.polygon(np.stack([seg_x, seg_y], axis=1).reshape([-1]).tolist(), fill='#010101')
                # plt.imshow(image)
                # plt.show()
                sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
                kp = np.array(label['keypoints'])
                keypoints_array.append(kp)
            except KeyError:
                pass

        sample['keypoints'] = keypoints_array
        sample['segment'] = segment_array
        sample = Rescale(320)(sample)
        sample = RandomCrop(inputsize)(sample)
        sample = RandomHorizontalFlip()(sample)
        sample['image'] = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)(sample['image'])

        # Label_map_keypoints = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        # Label_map_keypoints = Image.fromarray(Label_map_keypoints, 'L')
        # Label_map_background = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        # Label_map_background = Image.fromarray(Label_map_background, 'L')
        # draw_keypoints = ImageDraw.Draw(Label_map_keypoints)
        # draw_background = ImageDraw.Draw(Label_map_background)
        #
        # draw = ImageDraw.Draw(sample['image'])
        # for i in range(len(sample['segment'])):
        #     segment = sample['segment'][i]
        #     seg_x = np.array(segment[0::2]).astype(np.int)
        #     seg_y = np.array(segment[1::2]).astype(np.int)
        #     draw_background.polygon(np.stack([seg_x, seg_y], axis=1).reshape([-1]).tolist(), fill='#010101')
        #     x = np.array(sample['keypoints'][i][0::3]).astype(np.int)
        #     y = np.array(sample['keypoints'][i][1::3]).astype(np.int)
        #     v = sample['keypoints'][i][2::3]
        #     for k in range(keypoints):
        #         if v[k] > 0:
        #             draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(sample['image'])
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(Label_map_background)
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(Label_map_keypoints)
        #     plt.show()
        #     print('esf')
        Label_map_skeleton = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_map_skeleton = Image.fromarray(Label_map_skeleton, 'L')
        Label_map_keypoints = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_map_keypoints = Image.fromarray(Label_map_keypoints, 'L')
        Label_map_background = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_map_background = Image.fromarray(Label_map_background, 'L')
        draw_skeleton = ImageDraw.Draw(Label_map_skeleton)
        draw_keypoints = ImageDraw.Draw(Label_map_keypoints)
        draw_background = ImageDraw.Draw(Label_map_background)
        Gauss_map = np.zeros([17, int(inputsize / 4), int(inputsize / 4)])

        for i in range(len(sample['segment'])):
            segment = sample['segment'][i]
            seg_x = np.array(segment[0::2]).astype(np.int)
            seg_y = np.array(segment[1::2]).astype(np.int)
            draw_background.polygon(np.stack([seg_x, seg_y], axis=1).reshape([-1]).tolist(), fill='#010101')
            x = np.array(sample['keypoints'][i][0::3]).astype(np.int)
            y = np.array(sample['keypoints'][i][1::3]).astype(np.int)
            v = sample['keypoints'][i][2::3]
            sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
            kp = np.array(label['keypoints'])
            for k in range(keypoints):
                if v[k] > 0:
                    # sigma = 1
                    # mask_x = np.matlib.repmat(x[k], int(inputsize / 4), int(inputsize / 4))
                    # mask_y = np.matlib.repmat(y[k], int(inputsize / 4), int(inputsize / 4))
                    #
                    # x1 = np.arange(int(inputsize / 4))
                    # x_map = np.matlib.repmat(x1, int(inputsize / 4), 1)
                    #
                    # y1 = np.arange(int(inputsize / 4))
                    # y_map = np.matlib.repmat(y1, int(inputsize / 4), 1)
                    # y_map = np.transpose(y_map)
                    #
                    # temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)
                    #
                    # Gauss_map[k, :, :] += np.exp(-temp)
                    draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
            for i, sk in enumerate(sks):
                if np.all(v[sk] > 0):
                    draw_skeleton.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                                       'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        del draw_keypoints, draw_skeleton, draw_background
        # plt.subplot(1, 4, 1)
        # plt.imshow(sample['image'])
        # plt.subplot(1, 4, 2)
        # plt.imshow(Label_map_background)
        # plt.subplot(1, 4, 3)
        # plt.imshow(Label_map_skeleton)
        # plt.subplot(1, 4, 4)
        # plt.imshow(Label_map_keypoints)
        # plt.show()

        # print('esf')
        image_after = self.transform(sample['image'])
        return image_after, torch.Tensor(
            np.array(Label_map_keypoints)).long(), torch.Tensor(
            np.array(Label_map_skeleton)).long(), torch.Tensor(
            np.array(Label_map_background)).long()


class Costomer_CrossEntropyLoss(loss._WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(Costomer_CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, fraction):
        if fraction < 0.1:
            fraction = 0.1
        loss = F.nll_loss(F.log_softmax(input), target, reduce=False)
        w, h = loss.shape[1:3]
        k = input.shape[2] * input.shape[3] * fraction
        loss, index = torch.topk(loss.view(input.shape[0], -1), int(k))
        # index_x, index_y = index % w, index / w
        # maps = np.zeros([w, h])
        # for i in range(len(index_x)):
        #     maps[index_y[i].cpu().numpy(), index_x[i].cpu().numpy()] = 1
        loss = loss.mean()
        return loss


class Costomer_CrossEntropyLoss_with_mask(loss._WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(Costomer_CrossEntropyLoss_with_mask, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, mask):
        loss = F.nll_loss(F.log_softmax(input), target, reduce=False)
        loss = torch.mul(loss, mask.float()).view([loss.shape[0], -1])
        loss = loss.mean()
        return loss


class Costomer_MSELoss_with_mask(loss._WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(Costomer_MSELoss_with_mask, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, mask):
        loss = F.mse_loss(input, target, reduce=False)
        loss = torch.mul(loss, mask.float().view([mask.shape[0], 1, mask.shape[1], mask.shape[2]])).view(
            [loss.shape[0], -1])
        loss = loss.mean()
        return loss


class Costomer_MSELoss(loss._WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(Costomer_MSELoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, fraction):
        if fraction < 0.25:
            fraction = 0.25
        loss = F.mse_loss(input, target, reduce=False)
        k = input.shape[2] * input.shape[3] * fraction
        loss, _ = torch.topk(loss.view(input.shape[0], -1), int(k))
        loss = loss.mean()
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, numIn, numOut, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.numIn = numIn
        self.numOut = numOut
        self.bn1 = nn.BatchNorm2d(numIn)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(numIn, int(numOut / 2), 1, 1)
        self.bn2 = nn.BatchNorm2d(int(numOut / 2))
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(int(numOut / 2), int(numOut / 2), 3, stride, 1)
        self.bn3 = nn.BatchNorm2d(int(numOut / 2))
        self.relu = nn.ReLU(True)
        self.conv3 = nn.Conv2d(int(numOut / 2), numOut, 1, 1)
        self.bn4 = nn.BatchNorm2d(numOut)
        self.downsaple = nn.Sequential(
            nn.Conv2d(numIn, numOut, 1, stride=stride, bias=False),
            nn.BatchNorm2d(numOut)
        )

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        out = self.bn4(x)
        if self.stride != 1 | self.numIn != self.numOut:
            residual = self.downsaple(residual)
        out += residual
        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP_Block(nn.Module):
    def __init__(self):
        super(ASPP_Block, self).__init__()
        inplanes = 256
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        out = self.conv1(x)
        return out


class hourglass_hourglass(nn.Module):
    def __init__(self, f):
        super(hourglass_hourglass, self).__init__()
        self.f = f

        self.downsample1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(f, f))
        self.downsample2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(f, f))
        self.downsample3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(f, f))
        self.downsample4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResidualBlock(f, f))

        self.residual1 = ResidualBlock(f, f)
        self.residual2 = ResidualBlock(f, f)
        self.residual3 = ResidualBlock(f, f)
        self.residual4 = ResidualBlock(f, f)
        self.residual5 = ResidualBlock(f, f)

        self.upsample1 = ResidualBlock(f, f)
        self.upsample2 = ResidualBlock(f, f)
        self.upsample3 = ResidualBlock(f, f)
        self.upsample4 = ResidualBlock(f, f)

    def forward(self, x):
        up1 = self.residual1(x)
        down1 = self.downsample1(x)
        up2 = self.residual2(down1)
        down2 = self.downsample2(down1)
        up3 = self.residual3(down2)
        down3 = self.downsample3(down2)
        up4 = self.residual4(down3)
        down4 = self.downsample4(down3)
        out = self.residual5(down4)
        out = self.upsample4(out)
        out = F.interpolate(out, scale_factor=2)
        out = out + up4
        out = self.upsample3(out)
        out = F.interpolate(out, scale_factor=2)
        out = out + up3
        out = self.upsample2(out)
        out = F.interpolate(out, scale_factor=2)
        out = out + up2
        out = self.upsample1(out)
        out = F.interpolate(out, scale_factor=2)
        out = out + up1
        return out


class hourglass(nn.Module):
    def __init__(self, f):
        super(hourglass, self).__init__()
        self.f = f

        self.downsample1 = ResidualBlock(f, f, stride=2)
        self.downsample2 = ResidualBlock(f, f, stride=2)
        self.downsample3 = ResidualBlock(f, f, stride=2)
        self.downsample4 = ResidualBlock(f, f, stride=2)

        self.residual1 = ResidualBlock(f, int(f / 2))
        self.residual2 = ResidualBlock(f, int(f / 2))
        self.residual3 = ResidualBlock(f, int(f / 2))
        self.residual4 = ResidualBlock(f, int(f / 2))

        self.upsample1 = ResidualBlock(f, int(f / 2))
        self.upsample2 = ResidualBlock(f, int(f / 2))
        self.upsample3 = ResidualBlock(f, int(f / 2))
        self.upsample4 = ResidualBlock(f, int(f / 2))

        self.aspp = ASPP_Block()

    def forward(self, x):
        up1 = self.residual1(x)
        down1 = self.downsample1(x)
        up2 = self.residual2(down1)
        down2 = self.downsample2(down1)
        up3 = self.residual3(down2)
        down3 = self.downsample3(down2)
        up4 = self.residual4(down3)
        down4 = self.downsample4(down3)
        out = self.aspp(down4)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample4(out)
        out = torch.cat([out, up4], dim=1)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample3(out)
        out = torch.cat([out, up3], dim=1)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample2(out)
        out = torch.cat([out, up2], dim=1)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample1(out)
        out = torch.cat([out, up1], dim=1)
        return out


class creatModel(nn.Module):
    def __init__(self):
        super(creatModel, self).__init__()
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, nFeats)
        )

        self.stage1 = hourglass(nFeats)
        self.stage1_out = nn.Conv2d(nFeats, nOutChannels_0, 1, 1, 0, bias=False)
        self.stage1_return = nn.Conv2d(nOutChannels_0, int(nFeats / 2), 1, 1, 0, bias=False)
        self.stage1_retuen_2 = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)
        self.stage1_down_feature = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)

        self.stage2 = hourglass(nFeats)
        self.stage2_out = nn.Conv2d(nFeats, nOutChannels_1, 1, 1, 0, bias=False)
        self.stage2_return = nn.Conv2d(nOutChannels_1, int(nFeats / 2), 1, 1, 0, bias=False)
        self.stage2_retuen_2 = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)
        self.stage2_down_feature = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)

        self.stage3 = hourglass(nFeats)
        self.stage3_out = nn.Conv2d(nFeats, nOutChannels_2, 1, 1, 0, bias=False)

    def forward(self, x):
        inter = self.preprocess1(x)
        out = []

        i = 0

        ll = self.stage1(inter)
        tmpOut = self.stage1_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage1_return(tmpOut)
        ll_ = self.stage1_retuen_2(ll)
        inter = self.stage1_down_feature(inter)
        inter = torch.cat([tmpOut, ll_, inter], dim=1)

        i = 1

        ll = self.stage2(inter)
        tmpOut = self.stage2_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage2_return(tmpOut)
        ll_ = self.stage2_retuen_2(ll)
        inter = self.stage2_down_feature(inter)
        inter = torch.cat([tmpOut, ll_, inter], dim=1)

        i = 2

        ll = self.stage3(inter)
        tmpOut = self.stage3_out(ll)
        out.insert(i, tmpOut)

        return out


class creatModel_hourglass(nn.Module):
    def __init__(self):
        super(creatModel_hourglass, self).__init__()
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, nFeats)
        )

        self.stage1 = nn.Sequential(
            hourglass_hourglass(nFeats),
            ResidualBlock(nFeats, nFeats),
            nn.Conv2d(nFeats, nFeats, 1, 1, 0),
            nn.BatchNorm2d(nFeats),
            nn.ReLU()
        )
        self.stage1_out = nn.Conv2d(nFeats, 16, 1, 1, 0, bias=False)
        self.stage1_return = nn.Conv2d(16, nFeats, 1, 1, 0, bias=False)
        self.stage1_down_feature = nn.Conv2d(nFeats, nFeats, 1, 1, 0, bias=False)

        self.stage2 = nn.Sequential(
            hourglass_hourglass(nFeats),
            ResidualBlock(nFeats, nFeats),
            nn.Conv2d(nFeats, nFeats, 1, 1, 0),
            nn.BatchNorm2d(nFeats),
            nn.ReLU()
        )
        self.stage2_out = nn.Conv2d(nFeats, 16, 1, 1, 0, bias=False)
        self.stage2_return = nn.Conv2d(16, nFeats, 1, 1, 0, bias=False)
        self.stage2_down_feature = nn.Conv2d(nFeats, nFeats, 1, 1, 0, bias=False)

        self.stage3 = nn.Sequential(
            hourglass_hourglass(nFeats),
            ResidualBlock(nFeats, nFeats),
            nn.Conv2d(nFeats, nFeats, 1, 1, 0),
            nn.BatchNorm2d(nFeats),
            nn.ReLU()
        )
        self.stage3_out = nn.Conv2d(nFeats, 16, 1, 1, 0, bias=False)
        self.stage3_return = nn.Conv2d(16, nFeats, 1, 1, 0, bias=False)
        self.stage3_down_feature = nn.Conv2d(nFeats, nFeats, 1, 1, 0, bias=False)

        self.stage4 = nn.Sequential(
            hourglass_hourglass(nFeats),
            ResidualBlock(nFeats, nFeats),
            nn.Conv2d(nFeats, nFeats, 1, 1, 0),
            nn.BatchNorm2d(nFeats),
            nn.ReLU()
        )
        self.stage4_out = nn.Conv2d(nFeats, 16, 1, 1, 0, bias=False)

    def forward(self, x):
        inter = self.preprocess1(x)
        out = []

        i = 0

        ll = self.stage1(inter)
        tmpOut = self.stage1_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage1_return(tmpOut)
        ll_ = self.stage1_down_feature(ll)
        inter = tmpOut + inter + ll_

        i = 1

        ll = self.stage2(inter)
        tmpOut = self.stage2_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage2_return(tmpOut)
        ll_ = self.stage2_down_feature(ll)
        inter = tmpOut + inter + ll_

        i = 2

        ll = self.stage3(inter)
        tmpOut = self.stage3_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage3_return(tmpOut)
        ll_ = self.stage3_down_feature(ll)
        inter = tmpOut + inter + ll_

        i = 3

        ll = self.stage4(inter)
        tmpOut = self.stage4_out(ll)
        out.insert(i, tmpOut)

        return out


class generateMask(nn.Module):
    def __init__(self):
        super(generateMask, self).__init__()
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, nFeats)
        )
        self.stage1 = hourglass(nFeats)

        self.stage1_out = nn.Conv2d(nFeats, nOutChannels_0, 1, 1, 0, bias=False)

    def forward(self, x):
        inter = self.preprocess1(x)
        ll = self.stage1(inter)
        tmpOut = self.stage1_out(ll)
        return tmpOut


class myImageDataset(data.Dataset):
    def __init__(self, setfile, imagedir, matdir, transform=None, dim=(256, 256), n_channels=3,
                 n_joints=16, mode='train'):
        'Initialization'
        T = scipy.io.loadmat(matdir, squeeze_me=True, struct_as_record=False)
        M = T['RELEASE']
        self.annots = M.annolist
        is_train = M.img_train
        label = M.act
        self.dim = dim
        self.imagedir = imagedir
        list_file = open(setfile, 'r')
        list = list_file.readlines()
        for i, name in enumerate(list):
            list[i] = name.split()[0]
        self.list = list
        self.n_channels = n_channels
        self.n_joints = n_joints
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        anno = self.annots[int(self.list[index])]
        image_name = anno.image.name
        image = Image.open(path.join(self.imagedir, image_name)).convert('RGB')
        w, h = image.size
        image = image.resize([inputsize, inputsize])

        rect = anno.annorect
        points_all = []

        points = rect.annopoints.point
        points_rect = np.zeros([nKeypoint_MPII, 3])
        for point in points:
            if point.is_visible == 0:
                is_visible = 0
            else:
                is_visible = 1
            points_rect[point.id] = [point.x, point.y, is_visible]

        Label_keypoints = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_keypoints = Image.fromarray(Label_keypoints, 'L')
        draw_keypoints = ImageDraw.Draw(Label_keypoints)

        Label_skeleton = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_skeleton = Image.fromarray(Label_skeleton, 'L')
        draw_skeleton = ImageDraw.Draw(Label_skeleton)

        xs = points_rect[:, 0] * inputsize / w / 4
        ys = points_rect[:, 1] * inputsize / h / 4
        v = points_rect[:, 2]

        for i in range(nKeypoint_MPII):
            if v[i] > 0:
                size = 1
                xs_low, ys_low, xs_high, ys_high = xs[i] - size / 2, ys[i] - size / 2, xs[i] + size / 2, ys[
                    i] + size / 2
                draw_keypoints.ellipse((xs_low, ys_low, xs_high, ys_high),
                                       fill='rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        for i, sk in enumerate(sks):
            if np.all(v[sk]) > 0:
                draw_skeleton.line(np.stack([xs[sk], ys[sk]], axis=1).reshape([-1]).tolist(),
                                   'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))

        rect = anno.annorect
        rect = [rect.x1 * inputsize / w / 4, rect.y1 * inputsize / h / 4, rect.x2 * inputsize / w / 4,
                rect.y2 * inputsize / h / 4]
        # cm = ScalarMappable(norm=Normalize(0, 15))
        # cm.set_array(np.array(Label_skeleton))
        # result = cm.to_rgba(np.array(Label_skeleton.resize([256, 256])))[:, :, :3]
        # plt.subplot(1, 3, 1)
        # plt.imshow(image)
        # plt.subplot(1, 3, 2)
        # plt.imshow(np.array(Label_keypoints))
        # plt.subplot(1, 3, 3)
        # plt.imshow(np.array(Label_skeleton))
        # plt.show()
        # print('ef')

        return transforms.ToTensor()(image).float(), torch.Tensor(np.array(Label_keypoints)).long(), torch.Tensor(
            np.array(Label_skeleton)).long(), torch.Tensor(np.array(rect))

        # image = Image.open(path.join(self.imagedir, self.list[index])).convert('RGB')
        # w, h = image.size
        # image = image.resize([inputsize, inputsize])
        # if self.transform is not None:
        #     image = self.transform(image)
        #
        # number = int(self.list[index][2:6]) - 1
        # Gauss_map = np.zeros([14, int(inputsize / 4), int(inputsize / 4)])
        # for k in range(14):
        #     xs = self.mat['joints'][0][k][number] / w * inputsize / 4
        #     ys = self.mat['joints'][1][k][number] / h * inputsize / 4
        #     sigma = 1
        #     mask_x = np.matlib.repmat(xs, int(inputsize / 4), int(inputsize / 4))
        #     mask_y = np.matlib.repmat(ys, int(inputsize / 4), int(inputsize / 4))
        #
        #     x1 = np.arange(int(inputsize / 4))
        #     x_map = np.matlib.repmat(x1, int(inputsize / 4), 1)
        #
        #     y1 = np.arange(int(inputsize / 4))
        #     y_map = np.matlib.repmat(y1, int(inputsize / 4), 1)
        #     y_map = np.transpose(y_map)
        #
        #     temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)
        #
        #     Gauss_map[k, :, :] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-temp)
        #
        # return image, torch.Tensor(Gauss_map)


# def COCO_to_LSP(input):
#     result = torch.zeros(input.shape[0], 14, input.shape[2], input.shape[3])
#     result[:, 0, :, :] = input[:, 16, :, :]
#     result[:, 1, :, :] = input[:, 14, :, :]
#     result[:, 2, :, :] = input[:, 12, :, :]
#     result[:, 3, :, :] = input[:, 11, :, :]
#     result[:, 4, :, :] = input[:, 13, :, :]
#     result[:, 5, :, :] = input[:, 15, :, :]
#     result[:, 6, :, :] = input[:, 10, :, :]
#     result[:, 7, :, :] = input[:, 8, :, :]
#     result[:, 8, :, :] = input[:, 6, :, :]
#     result[:, 9, :, :] = input[:, 5, :, :]
#     result[:, 10, :, :] = input[:, 7, :, :]
#     result[:, 11, :, :] = input[:, 9, :, :]
#     result[:, 12, :, :] = torch.mul(input[:, 5, :, :] + input[:, 6, :, :], 0.5)
#     result[:, 13, :, :] = input[:, 0, :, :]
#
#     return result


class PCKh(nn.Module):
    def __init__(self):
        super(PCKh, self).__init__()

    def forward(self, x, target, rect):
        accuracy = []
        predicts = []
        labels = []
        for i in range(x.shape[0]):
            correct = 0
            total = 0
            predict = np.zeros([x.shape[1], 2])
            label = np.zeros([x.shape[1], 2])
            standard = np.sqrt((rect[i][0] - rect[i][2]) ** 2 + (rect[i][1] - rect[i][3]) ** 2) * 0.6
            for j in range(x.shape[1]):
                try:
                    label_ys, label_xs = torch.nonzero(target[i] == (j + 1))[0]
                except:
                    continue
                predict_ys, predict_xs = torch.nonzero(x[i, j + 1, :, :] >= torch.max(x[i, j + 1, :, :]))[0]

                if torch.sqrt(
                        (torch.pow(label_ys - predict_ys, 2) + torch.pow(label_xs - predict_xs,
                                                                         2)).float()) < standard * 0.5:
                    correct += 1
                total += 1
                predict[j] = [predict_xs, predict_ys]
                label[j] = [label_xs, label_ys]
            accuracy.append(correct / total)
            predicts.append(predict)
            labels.append(label)
        return accuracy, predicts, labels


class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
        )

    def forward(self, x):
        inter = self.preprocess1(x)
        return inter


def main():
    if mode == 'train':
        if write:
            writer = SummaryWriter('runs/' + save_model_name)
        # generatemask = generateMask().cuda()
        model = creatModel()
        loss1_background = Costomer_CrossEntropyLoss().cuda()
        loss2_skeleton = Costomer_CrossEntropyLoss().cuda()
        loss3_keypoints = Costomer_CrossEntropyLoss().cuda()
        # pckh = PCKh()
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        if dataset == 'coco':
            imgLoader_train = data.DataLoader(
                myImageDataset_COCO(train_set_coco, train_image_dir_coco, transform=mytransform), batch_size=batch_size,
                shuffle=True, num_workers=16)
        elif dataset == 'mpii':
            image_dir = '/data/mpii/mpii_human_pose_v1/images'
            mat_dir = '/data/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
            train_set = 'mpii/train.txt'
            eval_set = 'mpii/eval.txt'
            imgLoader_train = data.DataLoader(myImageDataset(train_set, image_dir, mat_dir, mytransform),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=16)
            imgLoader_eval = data.DataLoader(myImageDataset(eval_set, image_dir, mat_dir, mytransform),
                                             batch_size=16,
                                             shuffle=True, num_workers=4)
            imgIter_eval = iter(imgLoader_eval)
        # imgLoader_eval = data.DataLoader(myImageDataset(image_dir, mat_dir, mytransform), 8, True, num_workers=16)
        # imgIter = iter(imgLoader_eval)
        # mask_opt = torch.optim.Adam(generatemask.parameters(), lr=1e-3, eps=1e-4)
        # generatemask, mask_opt = amp.initialize(generatemask, mask_opt, opt_level="O1")
        # generatemask.eval()

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
        model.cuda()
        model, opt = amp.initialize(model, opt, opt_level="O1")
        model.train()

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mask_opt, mode='min', patience=10)
        # if usemask:
        #     state = torch.load(load_mask_name)
        #     generatemask.load_state_dict(state['state_dict'])
        #     for p in generatemask.parameters():
        #         p.requires_grad = False
        # if train_mask or not os.path.isfile(load_mask_name):
        #     epoch = 0
        # else:
        #     state = torch.load(load_mask_name)
        #     generatemask.load_state_dict(state['state_dict'])
        #     mask_opt.load_state_dict(state['optimizer'])
        #     epoch = state['epoch']

        if retrain or not os.path.isfile(load_model_name):
            epoch = 0
        else:
            if fine_tune:
                pretrained_state = torch.load(load_model_name)
                pretrained_model_state: object = pretrained_state['state_dict']

                model_state = model.state_dict()
                fine_tune_model_state = {}
                for k in pretrained_model_state:
                    if pretrained_model_state[k].size() == model_state[k].size():
                        fine_tune_model_state[k] = pretrained_model_state[k]
                model_state.update(fine_tune_model_state)
                model.load_state_dict(model_state)
                epoch = 0
            else:
                state = torch.load(load_model_name)
                model.load_state_dict(state['state_dict'])
                opt.load_state_dict(state['optimizer'])
                epoch = state['epoch']

        while epoch <= epochs:
            for i, [x_, y_keypoints, y_skeleton, _] in enumerate(imgLoader_train, 0):
                bx_, by_keypoints, by_skeleton = x_.cuda().half(), y_keypoints.cuda(), y_skeleton.cuda()
                # bx_, by_keypoints, by_skeleton, by_background = x_.cuda(), y_keypoints.cuda(), y_skeleton.cuda(), y_background.cuda()
                # mask = generatemask(bx_)
                # mask_interpolate = F.interpolate(mask, scale_factor=4)
                # mask_interpolate = torch.argmax(mask_interpolate, dim=1).unsqueeze(1)
                # image_with_mask = torch.mul(bx_, mask_interpolate.half())
                result = model(bx_)
                # losses = loss_background.forward(mask, by_background, (epochs - epoch) / epochs)
                # loss_1 = loss1_background.forward(result[0], by_background, (epochs - epoch) / epochs)
                loss_2 = loss2_skeleton.forward(result[1], by_skeleton, (100 - epoch) / 100)
                loss_3 = loss3_keypoints.forward(result[2], by_keypoints, (100 - epoch) / 100)
                losses = loss_2 + loss_3
                opt.zero_grad()
                # mask_opt.zero_grad()
                # with amp.scale_loss(losses, mask_opt) as scaled_loss:
                #     scaled_loss.backward()
                with amp.scale_loss(losses, opt) as scaled_loss:
                    scaled_loss.backward()
                # losses.backward()
                # mask_opt.step()
                opt.step()
                if i % 50 == 0:
                    loss_record = losses.cpu().data.numpy()
                    # loss1_record = loss_1.cpu().data.numpy()
                    loss2_record = loss_2.cpu().data.numpy()
                    loss3_record = loss_3.cpu().data.numpy()
                    steps = i + len(imgLoader_train) * epoch
                    if write:
                        writer.add_scalar('Loss', loss_record, steps)
                        # writer.add_scalar('Loss_1', loss1_record, steps)
                        writer.add_scalar('Loss_2', loss2_record, steps)
                        writer.add_scalar('Loss_3', loss3_record, steps)

                    print('[{}/{}][{}/{}] Loss: {}'.format(
                        epoch, epochs, i, len(imgLoader_train), loss_record
                    ))
                if i % 100 == 0:
                    if write:
                        # try:
                        #     x_, y_keypoints, _, _, rect = imgIter_eval.next()
                        # except:
                        #     imgIter_eval = iter(imgLoader_eval)
                        #     x_, y_keypoints, _, _, rect = imgIter_eval.next()
                        #
                        # steps = i + len(imgLoader_train) * epoch
                        # bx_, by_keypoints, by_skeleton = x_.cuda().half(), y_keypoints.cuda(), y_skeleton.cuda()
                        # # bx_, by_background = x_.cuda(), y_background.cuda()
                        # mask = generatemask(bx_)
                        # mask_interpolate = F.interpolate(mask, scale_factor=4)
                        # mask_interpolate = torch.argmax(mask_interpolate, dim=1).unsqueeze(1)
                        # image_with_mask = torch.mul(bx_, mask_interpolate.half())
                        # result = model(image_with_mask)
                        image = torchvision.utils.make_grid(bx_, normalize=True, range=(0, 1))
                        mask = torch.nn.functional.softmax(result[0])
                        mask = torch.argmax(mask, dim=1).unsqueeze(1)
                        mask = torchvision.utils.make_grid(mask, normalize=True, range=(0, 1))
                        # object = torch.argmax(result, dim=1).unsqueeze(1)
                        # image_with_mask = torchvision.utils.make_grid(image_with_mask, normalize=True, range=(0, 1))
                        # mask = torch.argmax(mask, dim=1).unsqueeze(1)
                        cm = ScalarMappable(Normalize(0, 20))
                        skeleton = torch.nn.functional.softmax(result[1])
                        skeleton = torch.argmax(skeleton, dim=1)
                        skeleton = torch.Tensor(
                            cm.to_rgba(np.array(skeleton.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2,
                                                                                                                      3))
                        skeleton = torchvision.utils.make_grid(skeleton, normalize=True, range=(0, 1))
                        keypoints = torch.nn.functional.softmax(result[2])
                        keypoints = torch.argmax(keypoints[:, :, :, :], dim=1)
                        keypoints = torch.Tensor(
                            cm.to_rgba(np.array(keypoints.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(
                                2,
                                3))
                        keypoints = torchvision.utils.make_grid(keypoints, normalize=True, range=(0, 1))

                        writer.add_image('image', image, steps)
                        writer.add_image('mask', mask, steps)
                        writer.add_image('skeleton', skeleton, steps)
                        writer.add_image('keypoints', keypoints, steps)
                # if i % 100 == 0:
                #     model.eval()
                #     steps = i + len(imgLoader_train_coco) * epoch
                #     try:
                #         x_, y = imgIter.next()
                #     except StopIteration:
                #         imgIter = iter(imgLoader_eval)
                #         x_, y = imgIter.next()
                #     bx_, by = x_.cuda(), y.cuda()
                #     result = model(bx_)
                #
                #     results = result[0]
                #     mask = torch.argmax(results[0, :, :, :], dim=0)
                #     result_ = COCO_to_LSP(result[2])
                #     accuracy = pckh(result_, by)
                #     writer.add_scalar('accuracy', accuracy, steps)
                #     model.train()
            # scheduler.step(loss)
            epoch += 1
            # state = {
            #     'epoch': epoch,
            #     'state_dict': generatemask.state_dict(),
            #     'optimizer': mask_opt.state_dict(),
            # }
            # torch.save(state, save_mask_name)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(state, save_model_name)

    elif mode == 'test':
        # test_model = testModel()
        # stat(test_model, (3, 256, 256))
        # for parameter in test_model.parameters():
        #     if parameter.requires_grad:
        #         print(parameter.numel())
        # generatemask = generateMask().cuda().half().eval()
        # test_model = testModel()
        # stat(test_model, (3, 256, 256))
        model = creatModel()
        stat(model, (3, 256, 256))
        model_hourglass = creatModel_hourglass()
        stat(model_hourglass, (3, 256, 256))
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # state = torch.load(load_mask_name)
        # generatemask.load_state_dict(state['state_dict'])
        state = torch.load(load_model_name)
        model.load_state_dict(state['state_dict'])


        for parameter in model.parameters():
            print(parameter)

        # loss_background = Costomer_CrossEntropyLoss().cuda()
        test_mode = 'mpii'
        if test_mode == 'coco':

            imgLoader_val_coco = data.DataLoader(
                myImageDataset_COCO(val_set_coco, val_image_dir_coco, transform=mytransform), batch_size=8,
                shuffle=True, num_workers=1)
            for i, [x_, _, y_skeleton, y_background] in enumerate(imgLoader_val_coco):
                bx_, by_background = x_.cuda().half(), y_background.cuda()
                mask = generatemask.forward(bx_)
                # loss, maps = loss_background(mask, by_background, 0.75)
                # loss, maps_1 = loss_background(mask, by_background, 0.50)
                # loss, maps_2 = loss_background(mask, by_background, 0.25)

                # plt.subplot(2, 2, 1)
                # plt.imshow(transforms.ToPILImage()(x_[0]))
                # plt.subplot(2, 2, 2)
                # plt.imshow(maps)
                # plt.subplot(2, 2, 3)
                # plt.imshow(maps_1)
                # plt.subplot(2, 2, 4)
                # plt.imshow(maps_2)
                # plt.show()
                # print('sefe')
                mask_interpolate = F.interpolate(mask, scale_factor=4)
                mask_interpolate = torch.argmax(mask_interpolate, dim=1).unsqueeze(1)
                # for i in range(mask_interpolate.shape[0]):
                #     image_after = torch.mul(bx_[i, :, :, :], mask_interpolate[i, :, :, :].half())
                #     # x_ = np.array(bx_[i, :, :, :].cpu())
                #     # mask_interpolate_inter = np.array(mask_interpolate[i, :, :, :].cpu())
                #     # image_after = np.multiply(x_, mask_interpolate_inter)
                #     # image_after = torch.matmul(bx_[i, :, :, :], mask_interpolate[i, 0, :, :].half())
                #     # image_after = transforms.ToPILImage()(image_after.cpu().float())
                #     image_after = transforms.ToPILImage()(image_after.cpu().float())
                #     plt.imshow(image_after)
                #     print('esfes')
                image_with_mask = torch.mul(bx_, mask_interpolate.half())
                result = model(image_with_mask)

                # results = torch.argmax(result[1], dim=1)
                # for i in range(results.shape[0]):
                #     result_inter = results[i]
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(np.array(y_skeleton[i]))
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(np.array(result_inter.cpu()))
                #     plt.show()
                #
                # print('efef')
                # for i in range(result[1].shape[0]):
                #     result_every = result[0][i]
                #     for j in range(result_every.shape[0]):
                #         result_inter = result_every[j]
                #         plt.subplot(3, 10, j + 1)
                #         plt.imshow(result_inter.cpu().data.float().numpy())
                #     plt.subplot(3, 1, 3)
                #     plt.imshow(transforms.ToPILImage()(image_with_mask[i].cpu().float()))
                #     plt.show()
                #     print('esfe')
                for i in range(result[1].shape[0]):
                    image = transforms.ToPILImage()(x_[i])
                    image_draw = ImageDraw.Draw(image)
                    keypoints_every = result[1][i]
                    for j in range(keypoints_every.shape[0] - 1):
                        result_inter = keypoints_every[j + 1]
                        xs, ys = torch.nonzero(result_inter >= torch.max(result_inter))[0]
                        if result_inter[xs, ys] > threshold:
                            image_draw.point([xs * 4, ys * 4], fill='rgb({}, {}, {})'.format(j + 1, j + 1, j + 1))
                        plt.subplot(3, 10, j + 1)
                        plt.imshow(result_inter.cpu().data.float().numpy())
                    plt.subplot(3, 1, 3)
                    plt.imshow(transforms.ToPILImage()(image_with_mask[i].cpu().float()))
                    plt.show()
                    print('sefesf')

                # image = torchvision.utils.make_grid(bx_, normalize=True, range=(0, 1))
                # image = transforms.ToPILImage()(image.cpu().float())
                # object = torch.argmax(mask, dim=1).unsqueeze(1)
                # object = torchvision.utils.make_grid(object, normalize=True, range=(0, 1))
                # object = transforms.ToPILImage()(object.cpu().float())
                # image_with_mask = torchvision.utils.make_grid(image_with_mask, normalize=True, range=(0, 1))
                # image_with_mask = transforms.ToPILImage()(image_with_mask.cpu().float())
                # plt.subplot(2, 1, 1)
                # plt.imshow(image)
                # plt.subplot(2, 1, 2)
                # plt.imshow(image_with_mask)
                # plt.show()
                mask = torch.argmax(mask, dim=1).unsqueeze(1)
                skeleton = torch.mul(result[0], mask.half())
                cm = ScalarMappable(norm=Normalize(0, 20))
                skeleton = torch.argmax(skeleton[:, 1:, :, :], dim=1)
                skeleton = torch.Tensor(
                    cm.to_rgba(np.array(skeleton.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
                skeleton = torchvision.utils.make_grid(skeleton, normalize=True, range=(0, 1))
                skeleton = transforms.ToPILImage()(skeleton)
                skeleton_all = torch.argmax(result[0][:, 1:, :, :], dim=1)
                skeleton_all = torch.Tensor(
                    cm.to_rgba(np.array(skeleton_all.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
                skeleton_all = torchvision.utils.make_grid(skeleton_all, normalize=True, range=(0, 1))
                skeleton_all = transforms.ToPILImage()(skeleton_all)
                keypoints = torch.mul(result[1], mask.half())
                cm = ScalarMappable(norm=Normalize(0, 20))
                keypoints = torch.argmax(keypoints[:, 1:, :, :], dim=1)
                keypoints = torch.Tensor(
                    cm.to_rgba(np.array(keypoints.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
                keypoints = torchvision.utils.make_grid(keypoints, normalize=True, range=(0, 1))
                keypoints = transforms.ToPILImage()(keypoints)

                plt.subplot(4, 1, 1)
                plt.imshow(image)
                plt.subplot(4, 1, 2)
                plt.imshow(image_with_mask)
                plt.subplot(4, 1, 3)
                plt.imshow(keypoints)
                plt.subplot(4, 1, 4)
                plt.imshow(skeleton)
                plt.show()
                print('easfeswf')

        if test_mode == 'mpii':
            image_dir = '/data/mpii/mpii_human_pose_v1/images'
            mat_dir = '/data/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
            eval_set = 'mpii/eval.txt'
            test_set = 'mpii/test.txt'
            pckh = PCKh().cuda()
            imgLoader_eval = data.DataLoader(myImageDataset(test_set, image_dir, mat_dir, mytransform),
                                             batch_size=8,
                                             shuffle=True, num_workers=1)
            accryacy_every = np.array([])
            for i, [x_, y_keypoints, _, rect] in enumerate(imgLoader_eval):
                bx_ = x_.cuda().half()
                result = model(bx_)

                accuracy, pred, label = pckh.forward(nn.functional.softmax(result[2]), y_keypoints, rect)
                accryacy_every = np.append(accryacy_every, accuracy, 0)
                #
                # print('esf')
                # results = torch.argmax(result[1], dim=1)
                # for i in range(results.shape[0]):
                #     result_inter = results[i]
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(np.array(y_skeleton[i]))
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(np.array(result_inter.cpu()))
                #     plt.show()
                # # #
                # # # print('efef')
                # # # for i in range(result[1].shape[0]):
                # # #     result_every = result[0][i]
                # # #     for j in range(result_every.shape[0]):
                # # #         result_inter = result_every[j]
                # # #         plt.subplot(3, 10, j + 1)
                # # #         plt.imshow(result_inter.cpu().data.float().numpy())
                # # #     plt.subplot(3, 1, 3)
                # # #     plt.imshow(transforms.ToPILImage()(image_with_mask[i].cpu().float()))
                # # #     plt.show()
                # # #     print('esfe')
                # for i in range(len(pred)):
                #     image = transforms.ToPILImage()(x_[i])
                #     image = image.resize([rect[i][4], rect[i][5]])
                #     label_image = Image.fromarray(np.zeros_like(np.array(image)))
                #     image_draw = ImageDraw.Draw(image)
                #     label_draw = ImageDraw.Draw(label_image)
                #     # Label_keypoints = np.zeros([64,64])
                #     # Label_keypoints = Image.fromarray(Label_keypoints, 'L')
                #     # draw_keypoints = ImageDraw.Draw(Label_keypoints)
                #     for j in range(pred[i].shape[0]):
                #         xs, ys = pred[i][j]
                #         x_label, y_label = label[i][j]
                #         if not(xs == 0 and ys == 0):
                #             size = 2
                #             xs_low = xs - size/2
                #             ys_low = ys - size/2
                #             xs_high = xs + size/2
                #             ys_high = ys + size/2
                #             # image_draw.ellipse((xs_low, ys_low, xs_high, ys_high), fill='rgb({}, {}, {})'.format(255, 0, 0))
                #             image_draw.point((xs, ys), fill='rgb(255, 0, 0)')
                #             x_label_low = x_label - size / 2
                #             y_label_low = y_label - size / 2
                #             x_label_high = x_label + size / 2
                #             y_label_high = y_label + size / 2
                #             label_draw.point((x_label, y_label), fill='rgb(255,0,0)')
                #             # label_draw.ellipse((x_label_low, y_label_low, x_label_high, y_label_high),
                #             #                    fill='rgb({}, {}, {})'.format(255, 0, 0))
                #
                #             # draw_keypoints.point([xs, ys], fill='rgb({}, {}, {})'.format(j+1, j+1, j+1))
                #     # x1, y1, x2, y2 = rect[i]
                #     # image_draw.rectangle((x1 * 4, y1 * 4, x2 * 4, y2 * 4), outline='rgb(0, 255, 0)')
                #     for j, sk in enumerate(sks):
                #         if np.all(label[i][sk]) > 0:
                #             label_draw.line((label[i][sk][0][0], label[i][sk][1][0], label[i][sk][0][1], label[i][sk][1][1]),
                #                                'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(image)
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(label_image)
                #     plt.show()
                #     print(accuracy[i])
                #     print('sefesf')

                # # image = torchvision.utils.make_grid(bx_, normalize=True, range=(0, 1))
                # # image = transforms.ToPILImage()(image.cpu().float())
                # # object = torch.argmax(mask, dim=1).unsqueeze(1)
                # # object = torchvision.utils.make_grid(object, normalize=True, range=(0, 1))
                # # object = transforms.ToPILImage()(object.cpu().float())
                # # image_with_mask = torchvision.utils.make_grid(image_with_mask, normalize=True, range=(0, 1))
                # # image_with_mask = transforms.ToPILImage()(image_with_mask.cpu().float())
                # # plt.subplot(2, 1, 1)
                # # plt.imshow(image)
                # # plt.subplot(2, 1, 2)
                # # plt.imshow(image_with_mask)
                # # plt.show()
                # mask = torch.argmax(mask, dim=1).unsqueeze(1)
                # skeleton = torch.mul(result[0], mask.half())
                # cm = ScalarMappable(norm=Normalize(0, 20))
                # skeleton = torch.argmax(skeleton[:, 1:, :, :], dim=1)
                # skeleton = torch.Tensor(
                #     cm.to_rgba(np.array(skeleton.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
                # skeleton = torchvision.utils.make_grid(skeleton, normalize=True, range=(0, 1))
                # skeleton = transforms.ToPILImage()(skeleton)
                # skeleton_all = torch.argmax(result[0][:, 1:, :, :], dim=1)
                # skeleton_all = torch.Tensor(
                #     cm.to_rgba(np.array(skeleton_all.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
                # skeleton_all = torchvision.utils.make_grid(skeleton_all, normalize=True, range=(0, 1))
                # skeleton_all = transforms.ToPILImage()(skeleton_all)
                # keypoints = torch.mul(result[1], mask.half())
                # cm = ScalarMappable(norm=Normalize(0, 20))
                # keypoints = torch.argmax(keypoints[:, 1:, :, :], dim=1)
                # keypoints = torch.Tensor(
                #     cm.to_rgba(np.array(keypoints.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
                # keypoints = torchvision.utils.make_grid(keypoints, normalize=True, range=(0, 1))
                # keypoints = transforms.ToPILImage()(keypoints)
                #
                # plt.subplot(4, 1, 1)
                # plt.imshow(image)
                # plt.subplot(4, 1, 2)
                # plt.imshow(image_with_mask)
                # plt.subplot(4, 1, 3)
                # plt.imshow(keypoints)
                # plt.subplot(4, 1, 4)
                # plt.imshow(skeleton)
                # plt.show()
                # print('easfeswf')

        elif test_mode == 'test':
            image = Image.open('test_img/im6.png').resize([256, 256])
            image_normalize = (mytransform(image)).unsqueeze(0).cuda().half()
            mask = generatemask.forward(image_normalize)
            mask_interpolate = F.interpolate(mask, scale_factor=4)
            mask_interpolate = torch.argmax(mask_interpolate, dim=1).unsqueeze(1)
            image_with_mask = torch.mul(image_normalize, mask_interpolate.half())
            result = model(image_with_mask)
            results = mask.cpu().float().data.numpy()
            plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)  # 调整子图间距
            # draw = ImageDraw.Draw(image)
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            results = np.argmax(results[0, :, :, :], axis=0)
            plt.imshow(results)
            plt.show()
            plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)
            results = result[0].cpu().float().data.numpy()
            for i in range(nOutChannels_1):
                plt.subplot(3, int(nOutChannels_1 / 2), i + 1)
                result_print = results[0, i, :, :]
                plt.imshow(result_print)
            plt.subplot(3, 1, 3)
            plt.imshow(image)
            plt.show()

            mask = torch.argmax(mask, dim=1).unsqueeze(1)
            skeleton = torch.mul(result[0], mask.half())
            cm = ScalarMappable(norm=Normalize(0, 20))
            skeleton = torch.argmax(skeleton[:, 1:, :, :], dim=1)
            skeleton = torch.Tensor(
                cm.to_rgba(np.array(skeleton.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
            skeleton = torchvision.utils.make_grid(skeleton, normalize=True, range=(0, 1))
            skeleton = transforms.ToPILImage()(skeleton)
            plt.imshow(skeleton)
            plt.show()
            plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)
            results = result[1].cpu().float().data.numpy()
            for i in range(nOutChannels_2):
                plt.subplot(3, 9, i + 1)
                # result_print = np.maximum(np.multiply(results[0, i, :, :], mask), 0)
                result_print = results[0, i, :, :]
                plt.imshow(result_print)
            plt.subplot(3, 1, 3)
            plt.imshow(image)
            plt.show()
            keypoints = torch.mul(result[1], mask.half())
            cm = ScalarMappable(norm=Normalize(0, 20))
            keypoints = torch.argmax(keypoints[:, 1:, :, :], dim=1)
            keypoints = torch.Tensor(
                cm.to_rgba(np.array(keypoints.unsqueeze(1).cpu()))[:, 0, :, :, :3].swapaxes(3, 1).swapaxes(2, 3))
            keypoints = torchvision.utils.make_grid(keypoints, normalize=True, range=(0, 1))
            keypoints = transforms.ToPILImage()(keypoints)
            plt.imshow(keypoints)
            plt.show(keypoints)
            print('esfesgt')

        print('yyy')


if __name__ == '__main__':
    main()
