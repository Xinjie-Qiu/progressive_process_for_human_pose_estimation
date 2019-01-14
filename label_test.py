import cv2
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

image = Image.open('/data/lsp_dataset/images/im0001.jpg').convert('RGB')
w, h = image.size
image = image.resize([256, 256])
draw = ImageDraw.Draw(image)
mat = scipy.io.loadmat('/data/lsp_dataset/joints.mat')
for k in range(14):
    xs = mat['joints'][0][k][0] / w * 256
    ys = mat['joints'][1][k][0] / h * 256
    width = 5
    draw.ellipse([xs - width, ys - width, xs + width, ys + width], fill=(0, 255, 0), outline=(255, 0, 0))
del draw
# plt.imshow(image)
# plt.show()
head_distance = np.sqrt(np.square(mat['joints'][0][12][0] - mat['joints'][0][13][0]) + np.square(mat['joints'][1][12][0] - mat['joints'][1][13][0]))
print('yyy')