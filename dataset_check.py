import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import scipy.io
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
from os import path
from numpy import matlib

matplotlib.use('TkAgg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# LSP dataset
# 0 Right ankle
# 1 Right knee
# 2 Right hip
# 3 Left hip
# 4 Left knee
# 5 Left ankle
# 6 Right wrist
# 7 Right elbow
# 8 Right shoulder
# 9 Left shoulder
# 10 Left elbow
# 11 Left wrist
# 12 Neck
# 13 Head top

# coco dataset
# [0 "nose",
# 1 "left_eye",
# 2 "right_eye",
# 3 "left_ear",
# 4 "right_ear",
# 5 "left_shoulder",
# 6 "right_shoulder",
# 7 "left_elbow",
# 8 "right_elbow",
# 9 "left_wrist",
# 10 "right_wrist",
# 11 "left_hip",
# 12 "right_hip",
# 13 "left_knee",
# 14 "right_knee",
# 15 "left_ankle",
# 16 "right_ankle"]

# coco -> lsp
# 0 -> 13
# 5 + 6 -> 12
# 9 -> 11
# 7 -> 10
# 5 -> 9
# 6 -> 8
# 8 -> 7
# 10 -> 6
# 15 -> 5
# 13 -> 4
# 11 -> 3
# 12 -> 2
# 14 -> 1
# 16 -> 0

nModules = 2
nFeats = 256
nStack = 4
nOutChannels = 18
epochs = 1000
batch_size = 16
keypoints = 17

inputsize = 320

mode = 'test'
save_model_name = 'params_3_coco.pkl'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
eval_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_val2017.json'

train_image_dir_coco = '/data/COCO/COCO2017/train2017'
eval_image_dir_coco = '/data/COCO/COCO2017/val2017'

# rootdir = '/data/lsp_dataset/images/'


class PCKh(nn.Module):
    def __init__(self):
        super(PCKh, self).__init__()

    def forward(self, x, target):
        correct = 0
        total = 0
        for i in range(batch_size):
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
        draw_skeleton = ImageDraw.Draw(Label_map_skeleton)
        Label_map_keypoints = np.zeros([64, 64])
        Label_map_keypoints = Image.fromarray(Label_map_keypoints, 'L')
        draw_keypoints = ImageDraw.Draw(Label_map_keypoints)
        draw = ImageDraw.Draw(image)
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

                    draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image)
                    plt.subplot(1, 2, 2)
                    plt.imshow(Label_map_keypoints)
                    plt.show()
                    print('sefes')
            for i, sk in enumerate(sks):
                if np.all(v[sk] > 0):
                    draw_skeleton.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                                       'rgb({}, {}, {})'.format(1, 1, 1))
        del draw_skeleton
        return image_after, torch.Tensor(np.array(Gauss_map)).long(), torch.Tensor(
            np.array(Label_map_skeleton)).long()


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


def main():
    pckh = PCKh()
    image_dir = '/data/lsp_dataset/images'
    mat_dir = '/data/lsp_dataset/joints.mat'
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test = myImageDataset_COCO(train_set_coco, train_image_dir_coco, mytransform)
    x, y, y1 = test.__getitem__(2)
    test_loader = data.DataLoader(myImageDataset(image_dir, mat_dir, mytransform), 16, True, num_workers=1)
    for step, [x, y_keypoints] in enumerate(test_loader, 0):
        pckh(x, y_keypoints)
        plt.subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(x[0].cpu().data))
        plt.subplot(1, 2, 2)
        plt.imshow(y_keypoints[0, 0, :, :].cpu().data.numpy())
        plt.show()
        print('efds')
    print('yyy')


if __name__ == '__main__':
    main()
