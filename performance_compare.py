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

load_model_name = 'params_1_add_cross_entropy_and_bootstrapped_together_fine_tune'
load_model_name_hourglass = 'params_3_fine_tune'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
val_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_val2017.json'
train_image_dir_coco = '/data/COCO/COCO2017/train2017/'
val_image_dir_coco = '/data/COCO/COCO2017/val2017'

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


class PCKh(nn.Module):
    def __init__(self):
        super(PCKh, self).__init__()

    def forward(self, x, target, rect):
        accuracy = np.zeros([x.shape[0], 11])
        predicts = []
        labels = []
        stand_dist = []
        for i in range(x.shape[0]):
            correct = np.zeros([11])
            total = np.zeros([11])
            predict = np.zeros([nKeypoint_MPII, 2])
            label = np.zeros([x.shape[1], 2])
            standard = np.sqrt((rect[i][0] - rect[i][2]) ** 2 + (rect[i][1] - rect[i][3]) ** 2) * 0.6
            for j in range(x.shape[1] - 1):
                try:
                    label_ys, label_xs = torch.nonzero(target[i] == (j + 1))[0]
                except:
                    continue
                predict_ys, predict_xs = torch.nonzero(x[i, j + 1, :, :] >= torch.max(x[i, j + 1, :, :]))[0]
                distance = torch.sqrt(
                        (torch.pow(label_ys - predict_ys, 2) + torch.pow(label_xs - predict_xs,
                                                                         2)).float()) / standard
                for step, k in enumerate(np.arange(0, 0.55, 0.05)):
                    if distance < k:
                        correct[step] += 1
                    total[step] += 1
                predict[j] = [predict_xs, predict_ys]
                label[j] = [label_xs, label_ys]
            accuracy[i] = (correct / total)
            predicts.append(predict)
            labels.append(label)
            stand_dist.append(standard)
        return accuracy, predicts, labels, stand_dist


class PCKh_hourglass(nn.Module):
    def __init__(self):
        super(PCKh_hourglass, self).__init__()

    def forward(self, x, target, rect):
        accuracy = np.zeros([x.shape[0], 11])
        predicts = []
        labels = []
        stand_dist = []
        for i in range(x.shape[0]):
            correct = np.zeros([11])
            total = np.zeros([11])
            predict = np.zeros([x.shape[1], 2])
            label = np.zeros([x.shape[1], 2])
            standard = np.sqrt((rect[i][0] - rect[i][2]) ** 2 + (rect[i][1] - rect[i][3]) ** 2) * 0.6
            for j in range(x.shape[1]):
                try:
                    label_ys, label_xs = torch.nonzero(target[i] == (j + 1))[0]
                except:
                    continue
                predict_ys, predict_xs = torch.nonzero(x[i, j, :, :] >= torch.max(x[i, j, :, :]))[0]
                distance = torch.sqrt(
                        (torch.pow(label_ys - predict_ys, 2) + torch.pow(label_xs - predict_xs,
                                                                         2)).float()) / standard
                for step, k in enumerate(np.arange(0, 0.55, 0.05)):
                    if distance < k:
                        correct[step] += 1
                    total[step] += 1
                predict[j] = [predict_xs, predict_ys]
                label[j] = [label_xs, label_ys]
            accuracy[i] = (correct / total)
            predicts.append(predict)
            labels.append(label)
            stand_dist.append(stand_dist)
        return accuracy, predicts, labels, stand_dist


if __name__ == '__main__':
    model = creatModel().half().cuda().eval()
    model_hourglass = creatModel_hourglass().half().cuda().eval()
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    state = torch.load(load_model_name)
    model.load_state_dict(state['state_dict'])
    state_hourglass = torch.load(load_model_name_hourglass)
    model_hourglass.load_state_dict(state_hourglass['state_dict'])
    image_dir = '/data/mpii/mpii_human_pose_v1/images'
    mat_dir = '/data/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    eval_set = 'mpii/eval.txt'
    test_set = 'mpii/test.txt'
    pckh = PCKh().cuda()
    pckh_hourglass = PCKh_hourglass().cuda()
    imgLoader_eval = data.DataLoader(myImageDataset(test_set, image_dir, mat_dir, mytransform),
                                     batch_size=8,
                                     shuffle=False, num_workers=8)
    accryacy_every = []
    accryacy_every_hourglass = []
    stand_every = []
    for i, [x_, y_keypoints, _, rect] in enumerate(imgLoader_eval):
        bx_ = x_.cuda().half()
        result = model(bx_)
        result_hourglass = model_hourglass(bx_)
        #
        accuracy, pred, label, standard = pckh.forward(nn.functional.softmax(result[2]), y_keypoints, rect)
        accuracy_hourglass, pred_hourglass, label_hourglass, standard_hourglass = pckh_hourglass.forward(nn.functional.softmax(result_hourglass[2]), y_keypoints, rect)
        stand_every.append(standard)
        # for i in range(len(pred)):
        #     ### small stand
        #     if standard[i] < 3 and accuracy[i].max() > 0.7:
        #         image = transforms.ToPILImage()(x_[i])
        #         image_show = image.resize([64, 64]).copy()
        #         image_hourglass = image_show.copy()
        #         image_draw = ImageDraw.Draw(image_show)
        #         image_draw_hourglass = ImageDraw.Draw(image_hourglass)
        #         cm = ScalarMappable(Normalize(0, nKeypoint_MPII - 1))
        #         RGB = cm.to_rgba(range(nKeypoint_MPII), bytes=True)[:, :3]
        #         for j in range(pred[i].shape[0]):
        #             xs, ys = pred[i][j]
        #             xs_hourglass, ys_hourglass = pred_hourglass[i][j]
        #             if not(xs == 0 and ys == 0):
        #                 image_draw.point((xs, ys), fill='rgb({},{},{})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #             if not(xs_hourglass == 0 and ys_hourglass == 0):
        #                 image_draw_hourglass.point((xs_hourglass, ys_hourglass),
        #                                            fill='rgb({},{},{})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #         cm = ScalarMappable(Normalize(0, nKeypoint_MPII - 1))
        #         RGB = cm.to_rgba(range(nKeypoint_MPII), bytes=True)[:, :3]
        #         # for j, sk in enumerate(sks):
        #         #     if np.all(pred[i][sk]) > 0:
        #         #         image_draw.line((pred[i][sk][0][0], pred[i][sk][0][1], pred[i][sk][1][0], pred[i][sk][1][1]),
        #         #                            'rgb({}, {}, {})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #         plt.imshow(image_show)
        #         # plt.show()
        #         print(accuracy[i])
        #         print('sefesf')
        #     ### big stand
        #     if standard[i] > 15 and accuracy[i].max() > 0.7:
        #         image = transforms.ToPILImage()(x_[i])
        #         image_show = image.resize([64, 64]).copy()
        #         image_hourglass = image_show.copy()
        #         image_draw = ImageDraw.Draw(image_show)
        #         image_draw_hourglass = ImageDraw.Draw(image_hourglass)
        #         cm = ScalarMappable(Normalize(0, nKeypoint_MPII - 1))
        #         RGB = cm.to_rgba(range(nKeypoint_MPII), bytes=True)[:, :3]
        #         for j in range(pred[i].shape[0]):
        #             xs, ys = pred[i][j]
        #             xs_hourglass, ys_hourglass = pred_hourglass[i][j]
        #             if not (xs == 0 and ys == 0):
        #                 image_draw.point((xs, ys), fill='rgb({},{},{})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #             if not(xs_hourglass == 0 and ys_hourglass == 0):
        #                 image_draw_hourglass.point((xs_hourglass, ys_hourglass),
        #                                            fill='rgb({},{},{})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #         cm = ScalarMappable(Normalize(0, nKeypoint_MPII - 1))
        #         RGB = cm.to_rgba(range(nKeypoint_MPII), bytes=True)[:, :3]
        #         # for j, sk in enumerate(sks):
        #         #     if np.all(pred[i][sk]) > 0:
        #         #         image_draw.line((pred[i][sk][0][0], pred[i][sk][0][1], pred[i][sk][1][0], pred[i][sk][1][1]),
        #         #                         'rgb({}, {}, {})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #         plt.imshow(image)
        #         plt.show()
        #         print(accuracy[i])
        #         print('sefesf')


        accryacy_every.append(accuracy)
        accryacy_every_hourglass.append(accuracy_hourglass)
        # ### compare with stacked hourglass networks
        # if (accuracy[0][10] - accuracy_hourglass[0][10] > 0.3):
        #     skeleton_result = nn.functional.interpolate(result[1], scale_factor=4)
        #     for i in range(len(pred)):
        #         image = transforms.ToPILImage()(x_[i])
        #         image = image.resize([64, 64])
        #         image_hourglass = image.copy()
        #         # label_image = Image.fromarray(np.zeros_like(np.array(image)))
        #         image_draw = ImageDraw.Draw(image)
        #         image_draw_hourglass = ImageDraw.Draw(image_hourglass)
        #         # label_draw = ImageDraw.Draw(label_image)
        #
        #         cm = ScalarMappable(Normalize(0, nKeypoint_MPII - 1))
        #         RGB = cm.to_rgba(range(nKeypoint_MPII), bytes=True)[:, :3]
        #         for j in range(pred[i].shape[0]):
        #             xs, ys = pred[i][j]
        #             xs_hourglass, ys_hourglass = pred_hourglass[i][j]
        #             x_label, y_label = label[i][j]
        #             if not(xs == 0 and ys == 0):
        #                 # size = 2
        #                 # xs_low = xs - size/2
        #                 # ys_low = ys - size/2
        #                 # xs_high = xs + size/2
        #                 # ys_high = ys + size/2
        #                 # image_draw.ellipse((xs_low, ys_low, xs_high, ys_high), fill='rgb({}, {}, {})'.format(255, 0, 0))
        #                 image_draw.point((xs, ys), fill='rgb({},{},{})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #                 # xs_hourglass_low = xs_hourglass - size / 2
        #                 # ys_hourglass_low = ys_hourglass - size / 2
        #                 # xs_hourglass_high = xs_hourglass + size / 2
        #                 # ys_hourglass_high = ys_hourglass + size / 2
        #                 image_draw_hourglass.point((xs_hourglass, ys_hourglass),
        #                                  fill='rgb({},{},{})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #                 # label_draw.ellipse((x_label_low, y_label_low, x_label_high, y_label_high),
        #                 #                    fill='rgb({}, {}, {})'.format(255, 0, 0))
        #
        #                 # draw_keypoints.point([xs, ys], fill='rgb({}, {}, {})'.format(j+1, j+1, j+1))
        #         cm = ScalarMappable(Normalize(0, nKeypoint_MPII - 1))
        #         RGB = cm.to_rgba(range(nKeypoint_MPII), bytes=True)[:, :3]
        #         # for j, sk in enumerate(sks):
        #         #     if np.all(pred[i][sk]) > 0:
        #         #         image_draw.line((pred[i][sk][0][0], pred[i][sk][0][1], pred[i][sk][1][0], pred[i][sk][1][1]),
        #         #                            'rgb({}, {}, {})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #         #     if np.all(pred_hourglass[i][sk]) > 0:
        #         #         image_draw_hourglass.line((pred[i][sk][0][0], pred[i][sk][0][1], pred[i][sk][1][0], pred[i][sk][1][1]),
        #         #                         'rgb({}, {}, {})'.format(RGB[j][0], RGB[j][1], RGB[j][2]))
        #         plt.subplot(1, 2, 1)
        #         plt.imshow(image)
        #         plt.subplot(1, 2, 2)
        #         plt.imshow(image_hourglass)
        #         plt.show()
        #         print(accuracy[i])
        #         print('sefesf')
    accryacy_every = np.concatenate(accryacy_every, axis=0)
    accuracy_mean = accryacy_every.mean(axis=0)
    print('esfes')

