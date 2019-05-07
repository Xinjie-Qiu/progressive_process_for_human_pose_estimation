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
import json
from pycocotools.coco import COCO
from os import path
from numpy import matlib
import random

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

inputsize = 256

mode = 'test'
save_model_name = 'params_3_coco.pkl'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
eval_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_val2017.json'

train_image_dir_coco = '/data/COCO/COCO2017/train2017'
eval_image_dir_coco = '/data/COCO/COCO2017/val2017'

# rootdir = '/data/lsp_dataset/images/'


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
        image = sample['image']
        image = transforms.ColorJitter(0.5, 0.5, 0.5, 0.3)(image)
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.subplot(1, 4, 2)
        plt.imshow(Label_map_background)
        plt.subplot(1, 4, 3)
        plt.imshow(Label_map_skeleton)
        plt.subplot(1, 4, 4)
        plt.imshow(Label_map_keypoints)
        plt.show()

        print('esf')
        image_after = self.transform(sample['image'])
        return image_after, torch.Tensor(np.array(Gauss_map)), torch.Tensor(
            np.array(Label_map_skeleton)).long(), torch.Tensor(
            np.array(Label_map_background)).long()


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


class myImageDataset(data.Dataset):
    def __init__(self, imagedir, matdir, transform=None, dim=(256, 256), n_channels=3,
                 n_joints=14):
        'Initialization'
        T = scipy.io.loadmat(matdir, squeeze_me=True, struct_as_record=False)
        M = T['RELEASE']
        annots = M.annolist
        is_train = M.img_train
        label = M.act
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
    # T = scipy.io.loadmat(mat_dir, squeeze_me=True, struct_as_record=False)
    # M = T['RELEASE']
    # annots = M.annolist
    # is_train = M.img_train
    # label = M.act
    # for i in range(len(annots)):
    #     # if is_train[i] == 0:
    #     img_name = annots[i].image
    #     points_fmted = []
    #     annot = annots[i]
    #     if 'annorect' in dir(annot):
    #         rects = annot.annorect
    #         if isinstance(rects, scipy.io.matlab.mio5_params.mat_struct):
    #             rects = np.array([rects])
    #             for rect in rects:
    #                 points_rect = []
    #                 try:
    #                     points = rect.annopoints.point
    #                 except:
    #                     continue
    #                 for point in points:
    #                     if point.is_visible in [0, 1]:
    #                         is_visible = point.is_visible
    #                     else:
    #                         is_visible = 0
    #                     points_rect.append((point.id, point.x, point.y, is_visible))
    #                     points_fmted.append(points_rect)

    # for aid, annot in enumerate(annots):
    #     img_name = annot.image.name
    #     points_fmted = []
    #     if 'annorect' in dir(annot):
    #         rects = annot.annorect
    #         if isinstance(rects, scipy.io.matlab.mio5_params.mat_struct):
    #             rects = np.array([rects])
    #             for rect in rects:
    #                 points_rect = []
    #                 try:
    #                     points = rect.annopoints.point
    #                 except:
    #                     continue
    #                 for point in points:
    #                     if point.is_visible in [0, 1]:
    #                         is_visible = point.is_visible
    #                     else:
    #                         is_visible = 0
    #                     points_rect.append((point.id, point.x, point.y, is_visible))
    #                     points_fmted.append(points_rect)



    image_dir = '/data/mpii/mpii_human_pose_v1/images'
    mat_dir = '/data/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # test = myImageDataset_COCO(train_set_coco, train_image_dir_coco, mytransform)
    # for i in range(100):
    #     x, y, y1 = test.__getitem__(0)
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
