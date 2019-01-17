import cv2
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

y1 = []
for i in range(10):
    y1.append(i)
y2 = np.linspace(0, 9, 10) * np.linspace(0, 9, 10)
x = np.linspace(0, 9, 10)
plt.plot(x, y1, y2)

plt.savefig('test.png')
plt.show()

# image = Image.open('/data/lsp_dataset/images/im0001.jpg').convert('RGB')
# w, h = image.size
# image = image.resize([256, 256])
# draw = ImageDraw.Draw(image)
# mat = scipy.io.loadmat('/data/lsp_dataset/joints.mat')
# for k in range(14):
#     xs = mat['joints'][0][k][0] / w * 256
#     ys = mat['joints'][1][k][0] / h * 256
#     width = 5
#     draw.ellipse([xs - width, ys - width, xs + width, ys + width], fill=(0, 255, 0), outline=(255, 0, 0))
# del draw
# plt.imshow(image)
# plt.show()
