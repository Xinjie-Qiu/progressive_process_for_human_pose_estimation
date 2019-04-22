import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import scipy.io
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
from os import path
import math
import torch.nn.functional as F
from scipy import ndimage
from numpy import matlib
from torch.optim import lr_scheduler
from graphviz import Digraph
from torch.autograd import Variable
from torchviz import make_dot

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0 "

nModules = 2
nFeats = 256
nStack = 3
nKeypoint = 17
nSkeleton = 19
nOutChannels_0 = 2
nOutChannels_1 = nSkeleton + 1
nOutChannels_2 = nKeypoint
epochs = 51
batch_size = 16
keypoints = 17
skeleton = 20

threshold = 0.8

mode = 'train'
save_model_name = 'params_2_aspp.pkl'

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


# def define_model(params):
#     def conv2d(input, params, base, stride=1, pad=0):
#         return F.conv2d(input, params[base + '.weight'],
#                         params[base + '.bias'], stride, pad)
#
#     def group(input, params, base, stride, n):
#         o = input
#         for i in range(0, n):
#             b_base = ('%s.block%d.conv') % (base, i)
#             x = o
#             o = conv2d(x, params, b_base + '0', pad=1, stride=i == 0 and stride or 1)
#             o = F.relu(o)
#             o = conv2d(o, params, b_base + '1', pad=1)
#             if i == 0 and stride != 1:
#                 o += F.conv2d(x, params[b_base + '_dim.weight'], stride=stride)
#             else:
#                 o += x
#             o = F.relu(o)
#         return o
#
#     # determine network size by parameters
#     blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
#                    for k in params.keys()]) for j in range(4)]
#
#     def f(input, params):
#         o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
#         o = F.relu(o)
#         o = F.max_pool2d(o, 3, 2, 1)
#         o_g0 = group(o, params, 'group0', 1, blocks[0])
#         o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
#         o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
#         o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
#         o = F.avg_pool2d(o_g3, 7, 1, 0)
#         o = o.view(o.size(0), -1)
#         o = F.linear(o, params['fc.weight'], params['fc.bias'])
#         return o
#
#     return f


# f = define_model(params)


class myImageDataset_COCO(data.Dataset):
    def __init__(self, anno, image_dir, transform=None):
        'Initialization'
        self.anno = COCO(anno)
        self.image_dir = image_dir
        self.lists = self.anno.getImgIds(catIds=self.anno.getCatIds())
        self.transform = transform

    def __len__(self):
        return len(self.lists)
        # return 100

    def __getitem__(self, index):
        list = self.lists[index]
        image_name = self.anno.loadImgs(list)[0]['file_name']
        image_path = path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        w, h = image.size
        image = image.resize([256, 256])
        if self.transform is not None:
            image_after = self.transform(image)
        label_id = self.anno.getAnnIds(list)
        labels = self.anno.loadAnns(label_id)
        Label_map_skeleton = np.zeros([64, 64])
        Label_map_skeleton = Image.fromarray(Label_map_skeleton, 'L')
        Label_map_background = np.zeros([64, 64])
        Label_map_background = Image.fromarray(Label_map_background, 'L')
        draw_skeleton = ImageDraw.Draw(Label_map_skeleton)
        draw_background = ImageDraw.Draw(Label_map_background)

        for label in labels:
            sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
            kp = np.array(label['keypoints'])
            x = np.array(kp[0::3] / w * 64).astype(np.int)
            y = np.array(kp[1::3] / h * 64).astype(np.int)
            v = kp[2::3]
            Gauss_map = np.zeros([17, 64, 64])
            for k in range(keypoints):
                if v[k] > 0:
                    sigma = 1
                    mask_x = np.matlib.repmat(x[k], 64, 64)
                    mask_y = np.matlib.repmat(y[k], 64, 64)

                    x1 = np.arange(64)
                    x_map = np.matlib.repmat(x1, 64, 1)

                    y1 = np.arange(64)
                    y_map = np.matlib.repmat(y1, 64, 1)
                    y_map = np.transpose(y_map)

                    temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

                    Gauss_map[k, :, :] = np.exp(-temp)
                    # draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
            for i, sk in enumerate(sks):
                if np.all(v[sk] > 0):
                    draw_background.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                                         'rgb({}, {}, {})'.format(1, 1, 1))
                    draw_skeleton.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                                       'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        del draw_skeleton, draw_background
        return image_after, torch.Tensor(np.array(Gauss_map)), torch.Tensor(
            np.array(Label_map_skeleton)).long(), torch.Tensor(
            np.array(Label_map_background)).long()


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


class ResidualBlock(nn.Module):
    def __init__(self, numIn, numOut, stride=1):
        super(ResidualBlock, self).__init__()
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
        out = self.conv3(x)
        return out


class hourglass(nn.Module):
    def __init__(self, n, f):
        super(hourglass, self).__init__()
        self.n = n
        self.f = f
        self.residual_block = ResidualBlock(f, f)
        self.residual_block_stride = ResidualBlock(f, f, stride=2)
        if n > 1:
            self.hourglass1 = hourglass(n - 1, f)
        self.maxpool = nn.MaxPool2d(2)
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
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv2 = nn.Conv2d(2 * f, f, 1, bias=False)
        self.conv3 = nn.Conv2d(f, f, 3, 2, 1)

    def forward(self, x):
        up1 = x

        up1 = self.residual_block(up1)
        low1 = self.residual_block_stride(x)
        if self.n > 1:
            low2 = self.hourglass1(low1)
        else:
            low2 = low1
        low3 = low2
        low3 = self.residual_block(low3)
        up2 = nn.functional.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)
        out = torch.cat([up1, up2], dim=1)
        out = self.conv2(out)
        return out


class lin(nn.Module):
    def __init__(self, numIn, numOut):
        super(lin, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.conv = nn.Conv2d(numIn, numOut, 1, 1, 0)
        self.bn = nn.BatchNorm2d(numOut)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class creatModel(nn.Module):
    def __init__(self):
        super(creatModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.relu = nn.ReLU()
        self.residual1 = ResidualBlock(64, 128, stride=2)
        self.residual2 = ResidualBlock(128, 128)
        self.residual3 = ResidualBlock(128, nFeats)
        self.hourglass1 = hourglass(4, nFeats)
        self.residual4 = ResidualBlock(nFeats, nFeats)
        self.lin = lin(nFeats, nFeats)
        self.conv2_0 = nn.Conv2d(nFeats, nOutChannels_0, 1, 1, 0, bias=False)
        self.conv4_0 = nn.Conv2d(nFeats + nOutChannels_0, nFeats, 1, 1, 0)
        self.conv2_1 = nn.Conv2d(nFeats, nOutChannels_1, 1, 1, 0, bias=False)
        self.conv4_1 = nn.Conv2d(nFeats + nOutChannels_1, nFeats, 1, 1, 0, bias=False)
        self.conv2_2 = nn.Conv2d(nFeats, nOutChannels_2, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)

        out = []
        inter = x
        for i in range(nStack):
            hg = self.hourglass1(inter)
            ll = hg
            ll = self.residual4(ll)
            ll = self.lin(ll)
            if i == 0:
                tmpOut = self.conv2_0(ll)
                out.insert(i, tmpOut)
                ll_ = torch.cat([ll, tmpOut], dim=1)
                inter = self.conv4_0(ll_)
            elif i == 1:
                tmpOut = self.conv2_1(ll)
                out.insert(i, tmpOut)
                ll_ = torch.cat([ll, tmpOut], dim=1)
                inter = self.conv4_1(ll_)
            elif i == 2:
                tmpOut = self.conv2_2(ll)
                out.insert(i, tmpOut)
        return out


def main():
    model = creatModel()
    x = torch.randn(1, 3, 256, 256)
    result = model(x)
    g = make_dot(result[2])
    g.view()


if __name__ == '__main__':
    main()
