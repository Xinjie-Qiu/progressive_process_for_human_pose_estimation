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
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
from os import path
import math
import torch.nn.functional as F
from scipy import ndimage
from numpy import matlib
from torch.optim import lr_scheduler
from apex import amp
import matplotlib
from torch.nn.modules import loss
from skimage.feature import peak_local_max
from tensorboardX import SummaryWriter
import random

matplotlib.use('TkAgg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nModules = 2
nFeats = 256
nStack = 3
nKeypoint = 17
nSkeleton = 19
nOutChannels_0 = 2
nOutChannels_1 = nSkeleton + 1
nOutChannels_2 = nKeypoint
epochs = 300
batch_size = 64
keypoints = 17
skeleton = 20
inputsize = 256

threshold = 0.8

mode = 'test'
load_model_name = 'params_3_mask_retrain'
save_model_name = 'params_3_mask_retrain'

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

sks = [[15, 13]
    , [13, 11]
    , [16, 14]
    , [14, 12]
    , [11, 12]
    , [5, 11]
    , [6, 12]
    , [5, 6]
    , [5, 7]
    , [6, 8]
    , [7, 9]
    , [8, 10]
    , [1, 2]
    , [0, 1]
    , [0, 2]
    , [1, 3]
    , [2, 4]
    , [3, 5]
    , [4, 6]]


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
                    sigma = 1
                    mask_x = np.matlib.repmat(x[k], int(inputsize / 4), int(inputsize / 4))
                    mask_y = np.matlib.repmat(y[k], int(inputsize / 4), int(inputsize / 4))

                    x1 = np.arange(int(inputsize / 4))
                    x_map = np.matlib.repmat(x1, int(inputsize / 4), 1)

                    y1 = np.arange(int(inputsize / 4))
                    y_map = np.matlib.repmat(y1, int(inputsize / 4), 1)
                    y_map = np.transpose(y_map)

                    temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

                    Gauss_map[k, :, :] += np.exp(-temp)
                    draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
            for i, sk in enumerate(sks):
                if np.all(v[sk] > 0):
                    draw_skeleton.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                                       'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        del draw_skeleton, draw_background
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
        return image_after, torch.Tensor(np.array(Gauss_map)), torch.Tensor(
            np.array(Label_map_skeleton)).long(), torch.Tensor(
            np.array(Label_map_background)).long()


class Costomer_CrossEntropyLoss(loss._WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(Costomer_CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, fraction):
        if fraction < 0.25:
            fraction = 0.25
        loss = F.nll_loss(F.log_softmax(input), target, reduce=False)
        k = input.shape[2] * input.shape[3] * fraction
        loss, _ = torch.topk(loss.view(input.shape[0], -1), int(k))
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


class hourglass(nn.Module):
    def __init__(self, f):
        super(hourglass, self).__init__()
        self.f = f

        self.downsample1 = ResidualBlock(f, f, stride=2)
        self.downsample2 = ResidualBlock(f, f, stride=2)
        self.downsample3 = ResidualBlock(f, f, stride=2)
        self.downsample4 = ResidualBlock(f, f, stride=2)

        self.residual1 = ResidualBlock(f, f)
        self.residual2 = ResidualBlock(f, f)
        self.residual3 = ResidualBlock(f, f)
        self.residual4 = ResidualBlock(f, f)

        self.upsample1 = ResidualBlock(f, f)
        self.upsample2 = ResidualBlock(f, f)
        self.upsample3 = ResidualBlock(f, f)
        self.upsample4 = ResidualBlock(f, f)

        self.conv4 = nn.Conv2d(2 * f, f, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(2 * f, f, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(2 * f, f, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(2 * f, f, 1, 1, bias=False)

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
        out = self.upsample4(out)
        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, up4], dim=1)
        out = self.conv4(out)
        out = self.upsample3(out)
        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, up3], dim=1)
        out = self.conv3(out)
        out = self.upsample2(out)
        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, up2], dim=1)
        out = self.conv2(out)
        out = self.upsample1(out)
        out = F.interpolate(out, scale_factor=2)
        out = torch.cat([out, up1], dim=1)
        out = self.conv1(out)
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
        self.stage1_down_feature = nn.Conv2d(nFeats, int(nFeats / 2), 1, 1, 0, bias=False)

        self.stage2 = hourglass(nFeats)
        self.stage2_out = nn.Conv2d(nFeats, nOutChannels_1, 1, 1, 0, bias=False)
        self.stage2_return = nn.Conv2d(nOutChannels_1, int(nFeats / 2), 1, 1, 0, bias=False)
        self.stage2_down_feature = nn.Conv2d(nFeats, int(nFeats / 2), 1, 1, 0, bias=False)

        self.stage3 = hourglass(nFeats)
        self.stage3_out = nn.Conv2d(nFeats, nOutChannels_2, 1, 1, 0, bias=False)

    def forward(self, x):
        i = 0
        inter = self.preprocess1(x)
        out = []
        ll = self.stage1(inter)
        tmpOut = self.stage1_out(ll)
        out.insert(i, tmpOut)

        tmpOut = self.stage1_return(tmpOut)
        inter = self.stage1_down_feature(inter)
        inter = torch.cat([tmpOut, inter], dim=1)

        i = 1


        ll = self.stage2(inter)
        tmpOut = self.stage2_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage2_return(tmpOut)
        inter = self.stage2_down_feature(inter)
        inter = torch.cat([tmpOut, inter], dim=1)

        i = 2

        ll = self.stage3(inter)
        tmpOut = self.stage3_out(ll)
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
    def __init__(self, imagedir, matdir, transform=None, dim=(256, 256), n_channels=3,
                 n_joints=14):
        'Initialization'
        self.mat = scipy.io.loadmat(matdir)
        self.dim = dim
        self.imagedir = imagedir
        self.list = os.listdir(imagedir)
        self.n_channels = n_channels
        self.n_joints = n_joints
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image = Image.open(path.join(self.imagedir, self.list[index])).convert('RGB')
        w, h = image.size
        image = image.resize([inputsize, inputsize])
        if self.transform is not None:
            image = self.transform(image)

        number = int(self.list[index][2:6]) - 1
        Gauss_map = np.zeros([14, int(inputsize / 4), int(inputsize / 4)])
        for k in range(14):
            xs = self.mat['joints'][0][k][number] / w * inputsize / 4
            ys = self.mat['joints'][1][k][number] / h * inputsize / 4
            sigma = 1
            mask_x = np.matlib.repmat(xs, int(inputsize / 4), int(inputsize / 4))
            mask_y = np.matlib.repmat(ys, int(inputsize / 4), int(inputsize / 4))

            x1 = np.arange(int(inputsize / 4))
            x_map = np.matlib.repmat(x1, int(inputsize / 4), 1)

            y1 = np.arange(int(inputsize / 4))
            y_map = np.matlib.repmat(y1, int(inputsize / 4), 1)
            y_map = np.transpose(y_map)

            temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

            Gauss_map[k, :, :] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-temp)

        return image, torch.Tensor(Gauss_map)


def COCO_to_LSP(input):
    result = torch.zeros(input.shape[0], 14, input.shape[2], input.shape[3])
    result[:, 0, :, :] = input[:, 16, :, :]
    result[:, 1, :, :] = input[:, 14, :, :]
    result[:, 2, :, :] = input[:, 12, :, :]
    result[:, 3, :, :] = input[:, 11, :, :]
    result[:, 4, :, :] = input[:, 13, :, :]
    result[:, 5, :, :] = input[:, 15, :, :]
    result[:, 6, :, :] = input[:, 10, :, :]
    result[:, 7, :, :] = input[:, 8, :, :]
    result[:, 8, :, :] = input[:, 6, :, :]
    result[:, 9, :, :] = input[:, 5, :, :]
    result[:, 10, :, :] = input[:, 7, :, :]
    result[:, 11, :, :] = input[:, 9, :, :]
    result[:, 12, :, :] = torch.mul(input[:, 5, :, :] + input[:, 6, :, :], 0.5)
    result[:, 13, :, :] = input[:, 0, :, :]

    return result


class PCKh(nn.Module):
    def __init__(self):
        super(PCKh, self).__init__()

    def forward(self, x, target):
        correct = 0
        total = 0
        for i in range(8):
            head_heat_map = target[i, 13, :, :]
            head_ys = torch.max(torch.max(head_heat_map, 1)[0], 0)[1]
            head_xs = torch.max(head_heat_map, 1)[1][head_ys]
            neck_heat_map = target[i, 1, :, :]
            neck_ys = torch.max(torch.max(neck_heat_map, 1)[0], 0)[1]
            neck_xs = torch.max(neck_heat_map, 1)[1][neck_ys]
            standard = torch.sqrt((torch.pow(head_ys - neck_ys, 2) + torch.pow(head_xs - neck_xs, 2)).float()) / 2
            for j in range(14):
                label_heat_map = target[i, j, :, :]
                if torch.max(label_heat_map) == 0:
                    continue
                label_ys = torch.max(torch.max(label_heat_map, 1)[0], 0)[1]
                label_xs = torch.max(label_heat_map, 1)[1][head_ys]
                predict_heat_map = x[i, j, :, :]
                predict_ys = torch.max(torch.max(predict_heat_map, 1)[0], 0)[1]
                predict_xs = torch.max(label_heat_map, 1)[1][head_ys]
                if torch.sqrt(
                        (torch.pow(label_ys - predict_ys, 2) + torch.pow(label_xs - predict_xs, 2)).float()) < standard:
                    correct += 1
                total += 1
        return correct / total


def main():
    if mode == 'train':
        image_dir = '/data/lsp_dataset/images'
        mat_dir = '/data/lsp_dataset/joints.mat'
        writer = SummaryWriter('runs/' + save_model_name)
        generatemask = generateMask().cuda()
        # model = creatModel()
        # model.cuda()
        loss_background = Costomer_CrossEntropyLoss().cuda()
        # loss2_skeleton = Costomer_CrossEntropyLoss_with_mask().cuda()
        # loss3_keypoints = Costomer_MSELoss_with_mask().cuda()
        pckh = PCKh()
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        imgLoader_train_coco = data.DataLoader(
            myImageDataset_COCO(train_set_coco, train_image_dir_coco, transform=mytransform), batch_size=batch_size,
            shuffle=True, num_workers=16)
        # imgLoader_eval = data.DataLoader(myImageDataset(image_dir, mat_dir, mytransform), 8, True, num_workers=16)
        # imgIter = iter(imgLoader_eval)
        mask_opt = torch.optim.Adam(generatemask.parameters(), lr=1e-3, eps=1e-4)
        # opt = torch.optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-4)
        generatemask, mask_opt = amp.initialize(generatemask, mask_opt, opt_level="O1")
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mask_opt, mode='min', patience=10)
        generatemask.train()
        # model, opt = amp.initialize(model, opt, opt_level="O1")
        # model.train()

        if retrain or not os.path.isfile(load_model_name):
            epoch = 0
        else:
            state = torch.load(load_model_name)
            generatemask.load_state_dict(state['state_dict'])
            mask_opt.load_state_dict(state['optimizer'])
            epoch = state['epoch']

        while epoch <= epochs:
            for i, [x_, _, _, y_background] in enumerate(imgLoader_train_coco, 0):
                bx_, by_background = x_.cuda(), y_background.cuda()
                result = generatemask(bx_)
                loss = loss_background.forward(result, by_background, (epochs - epoch) / epochs)
                # result = model(bx_)
                # loss_1 = loss1_background.forward(result[0], by_background)
                # loss_2 = loss2_skeleton.forward(result[1], by_skeleton, torch.argmax(result[0], dim=1))
                # loss_3 = loss3_keypoints.forward(result[2], by_keypoints, torch.argmax(result[0], dim=1))
                # losses = loss_1 + loss_2 + 100 * loss_3
                # opt.zero_grad()
                mask_opt.zero_grad()
                with amp.scale_loss(loss, mask_opt) as scaled_loss:
                    scaled_loss.backward()
                # with amp.scale_loss(losses, opt) as scaled_loss:
                #     scaled_loss.backward()
                # losses.backward()
                mask_opt.step()
                if i % 50 == 0:
                    loss_record = loss.cpu().data.numpy()
                    # loss1_record = loss_1.cpu().data.numpy()
                    # loss2_record = loss_2.cpu().data.numpy()
                    # loss3_record = loss_3.cpu().data.numpy()
                    steps = i + len(imgLoader_train_coco) * epoch
                    writer.add_scalar('Loss', loss_record, steps)
                    # writer.add_scalar('Loss_1', loss1_record, steps)
                    # writer.add_scalar('Loss_2', loss2_record, steps)
                    # writer.add_scalar('Loss_3', loss3_record, steps)

                    print('[{}/{}][{}/{}] Loss: {}'.format(
                        epoch, epochs, i, len(imgLoader_train_coco), loss_record
                    ))
                if i % 100 == 0:
                    steps = i + len(imgLoader_train_coco) * epoch
                    image = torchvision.utils.make_grid(bx_, normalize=True, range=(0, 1))
                    object = torch.argmax(result, dim=1).unsqueeze(1)
                    object = torchvision.utils.make_grid(object, normalize=True, range=(0, 1))
                    writer.add_image('image', image, steps)
                    writer.add_image('object', object, steps)
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
            state = {
                'epoch': epoch,
                'state_dict': generatemask.state_dict(),
                'optimizer': mask_opt.state_dict(),
            }
            torch.save(state, save_model_name)

    elif mode == 'test':
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        generatemask = generateMask().cuda().half().eval()
        state = torch.load(save_model_name)
        generatemask.load_state_dict(state['state_dict'])
        # model = creatModel()
        # model.eval().cuda().half()
        # state = torch.load(save_model_name)
        # model.load_state_dict(state['state_dict'])
        # epoch = state['epoch']
        test_mode = 'coco'
        if test_mode == 'coco':

            imgLoader_val_coco = data.DataLoader(
                myImageDataset_COCO(val_set_coco, val_image_dir_coco, transform=mytransform), batch_size=8,
                shuffle=True, num_workers=8)
            for i, [x_, _, _, y_background] in enumerate(imgLoader_val_coco):
                bx_ = x_.cuda().half()
                mask = generatemask.forward(bx_)
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
                image = torchvision.utils.make_grid(bx_, normalize=True, range=(0, 1))
                image = transforms.ToPILImage()(image.cpu().float())
                object = torch.argmax(mask, dim=1).unsqueeze(1)
                object = torchvision.utils.make_grid(object, normalize=True, range=(0, 1))
                object = transforms.ToPILImage()(object.cpu().float())
                image_with_mask = torchvision.utils.make_grid(image_with_mask, normalize=True, range=(0, 1))
                image_with_mask = transforms.ToPILImage()(image_with_mask.cpu().float())
                plt.subplot(3, 1, 1)
                plt.imshow(image)
                plt.subplot(3, 1, 2)
                plt.imshow(object)
                plt.subplot(3, 1, 3)
                plt.imshow(image_with_mask)
                plt.show()




        elif test_mode == 'test':
            image = Image.open('test_img/im6.png').resize([256, 256])
            image_normalize = (mytransform(image)).unsqueeze(0).cuda().half()
            result = generatemask.forward(image_normalize)
            # image = (image.cpu().float().numpy()[0].transpose((1, 2, 0)) * 255).astype('uint8')
            # image = Image.fromarray(image)

            results = result.cpu().float().data.numpy()
            plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)  # 调整子图间距
            # draw = ImageDraw.Draw(image)
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            mask = np.argmax(results[0, :, :, :], axis=0)
            plt.imshow(mask)
            plt.show()
            # plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)
            # results = result[1].cpu().float().data.numpy()
            # for i in range(nOutChannels_1):
            #     plt.subplot(3, int(nOutChannels_1 / 2), i + 1)
            #     result_print = results[0, i, :, :]
            #     plt.imshow(result_print)
            # plt.subplot(3, 1, 3)
            # plt.imshow(image)
            # plt.show()
            # plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)
            # results = result[2].cpu().float().data.numpy()
            # COCO_to_LSP(results)
            # for i in range(17):
            #     plt.subplot(3, 9, i + 1)
            #     # result_print = np.maximum(np.multiply(results[0, i, :, :], mask), 0)
            #     result_print = results[0, i, :, :]
            #
            #     peak_value = peak_local_max(result_print, min_distance=15)
            #
            #     y_point = peak_value[:, 0] * 4
            #     x_point = peak_value[:, 1] * 4
            #     plt.imshow(result_print)
            #
            #     width = 2
            #     for j in range(len(x_point)):
            #         draw.ellipse([x_point[j] - width, y_point[j] - width, x_point[j] + width, y_point[j] + width],
            #                      fill=(int(255 / 17 * i), int(255 / 17 * i), int(255 / 17 * i)),
            #                      outline=(int(255 / 17 * i), int(255 / 17 * i), int(255 / 17 * i)))
            #
            # del draw
            # plt.subplot(3, 1, 3)
            # plt.imshow(image)
            # plt.show()

        print('yyy')


if __name__ == '__main__':
    main()
