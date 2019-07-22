import torch
from torch.utils import data
import scipy
from scipy import io
from PIL import Image, ImageDraw
import os
from os import path
import numpy as np
from matplotlib import pyplot as plt

inputsize = 256

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


if __name__ == '__main__':
    image_dir = '/data/mpii/mpii_human_pose_v1/images'
    mat_dir = '/data/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
    T = scipy.io.loadmat(mat_dir, squeeze_me=True, struct_as_record=False)
    M = T['RELEASE']
    M = M
    annots = M.annolist
    is_train = M.img_train
    # is_test = is_train - 1
    lists = np.nonzero(is_train)
    single_person = np.zeros_like(annots)
    mult_person = np.zeros_like(annots)
    for i in lists[0]:
        anno = annots[i]
        rect = anno.annorect
        if isinstance(rect, scipy.io.matlab.mio5_params.mat_struct):
            try:
                points = rect.annopoints.point
                for point in points:
                    pass
                single_person[i] = 1
            except:
                pass
        else:
            can_be_use = True
            if len(rect) > 0:
                for j in range(len(rect)):
                    try:
                        points = rect[j].annopoints.point
                        for point in points:
                            pass
                    except:
                        can_be_use = False
                if can_be_use:
                    mult_person[i] = 1
    train_list = np.array([])
    eval_list = np.array([])
    test_list = np.array([])
    list = np.nonzero(single_person)[0]
    for i in range(len(list)):
        if i % 10 < 1:
            test_list = np.append(test_list, list[i])
        elif i % 10 < 3:
            eval_list = np.append(eval_list, list[i])
        else:
            train_list = np.append(train_list, list[i])
    test_writer = open('mpii/test.txt', 'w')
    eval_writer = open('mpii/eval.txt', 'w')
    train_writer = open('mpii/train.txt', 'w')
    for i in range(len(test_list)):
        test_writer.write(str(test_list[i].astype(np.int)) + '\n')
    for i in range(len(eval_list)):
        eval_writer.write(str(eval_list[i].astype(np.int)) + '\n')
    for i in range(len(train_list)):
        train_writer.write(str(train_list[i].astype(np.int)) + '\n')
    test_writer.close()
    eval_writer.close()
    train_writer.close()
    print('sefes')