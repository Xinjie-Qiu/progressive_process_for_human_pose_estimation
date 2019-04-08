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
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
eval_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_val2017.json'

train_image_dir_coco = '/data/COCO/COCO2017/train2017'
eval_image_dir_coco = '/data/COCO/COCO2017/val2017'

# rootdir = '/data/lsp_dataset/images/'


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
            image_after_transform = self.transform(image)
        label_id = self.anno.getAnnIds(list)
        labels = self.anno.loadAnns(label_id)
        Label_map = np.zeros([64, 64])
        Label_map = Image.fromarray(Label_map, 'L')
        draw = ImageDraw.Draw(Label_map)
        for label in labels:
            sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
            kp = np.array(label['keypoints'])
            x = np.array(kp[0::3] / w * 64).astype(np.int)
            y = np.array(kp[1::3] / h * 64).astype(np.int)
            v = kp[2::3]
            for i, sk in enumerate(sks):
                if np.all(v[sk] > 0):
                    draw.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                              'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))

        del draw
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(Label_map))
        plt.show()
        return image_after_transform, torch.Tensor(np.array(Label_map)).long()

def main():
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # test = myImageDataset_COCO(train_set_coco, train_image_dir_coco)
    test_loader = data.DataLoader(myImageDataset_COCO(train_set_coco, train_image_dir_coco, mytransform), 1, True, num_workers=1)
    for step, [x, y] in enumerate(test_loader, 0):
        print('efds')
    print('yyy')


if __name__ == '__main__':
    main()
