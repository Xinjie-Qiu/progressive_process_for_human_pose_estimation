from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor
from os import path
import numpy as np

keypoints = 17
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
train_image_dir_coco = '/data/COCO/COCO2017/train2017/'
anno = COCO(train_set_coco)
catIds = anno.getCatIds()
lists = anno.getImgIds(catIds=catIds)
for i in lists:
    image_name = anno.loadImgs(i)[0]['file_name']
    image_path = path.join(train_image_dir_coco, image_name)
    image = Image.open(image_path)
    image = image.convert('RGB')
    w, h = image.size
    image = image.resize([256, 256])
    plt.imshow(image)
    label_id = anno.getAnnIds(i)
    labels = anno.loadAnns(label_id)
    for label in labels:
        sks = np.array(anno.loadCats(label['category_id'])[0]['skeleton']) - 1
        kp = np.array(label['keypoints'])
        x = np.array(kp[0::3] / w * 64).astype(np.int)
        y = np.array(kp[1::3] / h * 64).astype(np.int)
        v = kp[2::3]
        Gauss_map = np.zeros([64, 64])
        Label_map = np.zeros([64, 64])
        Label_map = Image.fromarray(Label_map, 'L')
        draw = ImageDraw.Draw(Label_map)
        # for k in range(keypoints):
        #     if v[k] > 0:
        #         Gauss_map[x[k], y[k]] = k + 1
        #         plt.plot(x[k] * 4, y[k] * 4, linewidth=3, color='r')
        for i, sk in enumerate(sks):
            if np.all(v[sk] > 0):
                 draw.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(), 'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        # for sk in sks:
        #     if np.all(v[sk] > 0):
        #         plt.plot(x[sk], y[sk], linewidth=3, color='r')
        plt.show()
        # plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2)
        # plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
    print('sadf')
print('yyy')





