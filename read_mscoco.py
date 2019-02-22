import torch
import torch.nn as nn
import torch.utils.data as data
import json
import numpy as np
import numpy.matlib
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

keypoints = 17
image_dir = '/data/COCO2014/val2014/COCO_val2014_'
train_set_coco = '/data/COCO2014/annotations/person_keypoints_train2014.json'
eval_set_coco = '/data/COCO2014/annotations/person_keypoints_val2014.json'
train_image_dir_coco = '/data/COCO2014/train2014/COCO_train2014_'
eval_image_dir_coco = '/data/COCO2014/val2014/COCO_val2014_'

batch_size = 16
keypoints = 17

class myImageDataset_COCO(data.Dataset):
    def __init__(self, filename, image_dir, transform=None, dim=(256, 256), n_channels=3,
                 n_joints=keypoints):
        'Initialization'
        with open(filename) as f:
            datas = json.load(f)
        self.lists = datas['annotations']
        self.dim = dim
        file = open(filename)
        self.n_channels = n_channels
        self.n_joints = n_joints
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        data = self.lists[index]
        image_name = self.image_dir + '%012d.jpg' % data['image_id']
        image = Image.open(image_name)
        image = image.convert('RGB')
        w, h = image.size
        image = image.resize([256, 256])
        if self.transform is not None:
            image = self.transform(image)
        Gauss_map = np.zeros([keypoints, 64, 64])
        for k in range(keypoints):
            if data['keypoints'][k * 3 + 2] > 0:
                xs = data['keypoints'][k * 3] / w * 64
                ys = data['keypoints'][k * 3 + 1] / h * 64

                sigma = 1
                mask_x = np.matlib.repmat(xs, 64, 64)
                mask_y = np.matlib.repmat(ys, 64, 64)

                x1 = np.arange(64)
                x_map = np.matlib.repmat(x1, 64, 1)

                y1 = np.arange(64)
                y_map = np.matlib.repmat(y1, 64, 1)
                y_map = np.transpose(y_map)

                temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

                Gauss_map[k, :, :] = np.exp(-temp)

        return image, torch.Tensor(Gauss_map)


if __name__ == '__main__':
    dataset = myImageDataset_COCO(train_set_coco, train_image_dir_coco)
    image, result = dataset.__getitem__(5)
    print('yyy')
    draw = ImageDraw.Draw(image)
    for i in range(17):
        plt.subplot(3, 9, i + 1)
        plt.imshow(result[i, :, :])
    for i in range(17):
        x = result[i, :, :]
        if np.max(x.numpy() > 0.5):
            ys, xs = np.multiply(np.where(x == np.max(x.numpy())), 4)
            width = 5
            draw.ellipse([xs - width, ys - width, xs + width, ys + width], fill=(0, 255, 0), outline=(255, 0, 0))

    del draw
    plt.subplot(3, 1, 3)
    plt.imshow(image)
    plt.show()

    with open('/data/COCO2014/annotations/person_keypoints_val2014.json') as f:
        datas = json.load(f)
    for i in range(len(datas['annotations'])):
        data = datas['annotations'][i]
        image_name = image_dir + '%012d.jpg' % data['image_id']
        image = Image.open(image_name)
        w, h = image.size
        image = image.resize([256, 256])
        Gauss_map = np.zeros([keypoints, 64, 64])
        draw = ImageDraw.Draw(image)
        for k in range(keypoints):
            if data['keypoints'][k * 3 + 2] > 0:
                xs = data['keypoints'][k * 3] / w * 64
                ys = data['keypoints'][k * 3 + 1] / h * 64

                width = 1
                draw.ellipse([xs * 4 - width, ys * 4 - width, xs * 4 + width, ys * 4 + width], fill=(0, 255, 0), outline=(255, 0, 0))

                sigma = 1
                mask_x = np.matlib.repmat(xs, 64, 64)
                mask_y = np.matlib.repmat(ys, 64, 64)

                x1 = np.arange(64)
                x_map = np.matlib.repmat(x1, 64, 1)

                y1 = np.arange(64)
                y_map = np.matlib.repmat(y1, 64, 1)
                y_map = np.transpose(y_map)

                temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

                Gauss_map[k, :, :] = np.exp(-temp)
        del draw
        plt.imshow(image)
        plt.show()
        print('yyy')
    print(datas)
