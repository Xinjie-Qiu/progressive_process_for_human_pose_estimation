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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

nModules = 2
nFeats = 256
nStack = 4
nOutChannels = 18
epochs = 1000
batch_size = 16
keypoints = 17

mode = 'test'
save_model_name = 'params_3_coco.pkl'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO2014/annotations/person_keypoints_train2014.json'
eval_set_coco = '/data/COCO2014/annotations/person_keypoints_val2014.json'
train_image_dir_coco = '/data/COCO2014/train2014'
eval_image_dir_coco = '/data/COCO2014/val2014'

rootdir = '/data/lsp_dataset/images/'


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
            image = self.transform(image)
        label_id = self.anno.getAnnIds(list)
        labels = self.anno.loadAnns(label_id)
        Label_map_keypoints = np.zeros([64, 64])
        Label_map_keypoints = Image.fromarray(Label_map_keypoints, 'L')
        draw_keypoints = ImageDraw.Draw(Label_map_keypoints)
        Label_map_skeleton = np.zeros([64, 64])
        Label_map_skeleton = Image.fromarray(Label_map_skeleton, 'L')
        draw_skeleton = ImageDraw.Draw(Label_map_skeleton)
        for label in labels:
            sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
            kp = np.array(label['keypoints'])
            x = np.array(kp[0::3] / w * 64).astype(np.int)
            y = np.array(kp[1::3] / h * 64).astype(np.int)
            v = kp[2::3]
            for k in range(keypoints):
                if v[k] > 0:
                    draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
            for i, sk in enumerate(sks):
                if np.all(v[sk] > 0):
                    draw_skeleton.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                                       'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        del draw_keypoints, draw_skeleton
        # for label in labels:
        #     sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
        #     kp = np.array(label['keypoints'])
        #     x = np.array(kp[0::3] / w * 64).astype(np.int)
        #     y = np.array(kp[1::3] / h * 64).astype(np.int)
        #     v = kp[2::3]
        #     for k in range(keypoints):
        #         if v[k] > 0:
        #             draw.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
        # del draw
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(Label_map_skeleton.resize([256, 256])))
        plt.show()
        return image, torch.Tensor(np.array(Label_map_keypoints)).long(), torch.Tensor(
            np.array(Label_map_skeleton)).long()

def main():
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test = myImageDataset_COCO(train_set_coco, train_image_dir_coco)
    x, y = test.__getitem__(5)
    # imgLoader_train_coco = data.DataLoader(myImageDataset_COCO(train_set_coco, train_image_dir_coco, transform=mytransform), batch_size=batch_size, shuffle=True, num_workers=8)
    plt.imshow(x)
    plt.imshow(y.resize([256, 256]))
    plt.show()
    print('yyy')


if __name__ == '__main__':
    main()
