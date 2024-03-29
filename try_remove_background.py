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
from apex import amp
import matplotlib

matplotlib.use('TkAgg')

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
epochs = 50
batch_size = 32
keypoints = 17
skeleton = 20

threshold = 0.8

mode = 'test'
save_model_name = 'params_1_remove_background.pkl'

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
            try:
                segment = label['segmentation'][0]
                seg_x = np.multiply(segment[0::2], 64 / w)
                seg_y = np.multiply(segment[1::2], 64 / h)
                draw_background.polygon(np.stack([seg_x, seg_y], axis=1).reshape([-1]).tolist(), fill='#010101')
            except KeyError:
                pass
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
        self.stride = stride
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
        self.bn4 = nn.BatchNorm2d(numOut)
        self.downsaple = nn.Sequential(
            nn.Conv2d(numIn, numOut, 1, stride=stride, bias=False),
            nn.BatchNorm2d(numOut)
        )

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
        x = self.conv3(x)
        out = self.bn4(x)
        if self.stride != 1 | self.numIn != self.numOut:
            residual = self.downsaple(residual)
        out += residual
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
        self.conv4_0 = nn.Conv2d(2 * nFeats, nFeats, 1, 1, 0)
        self.conv2_1 = nn.Conv2d(nFeats, nOutChannels_1, 1, 1, 0, bias=False)
        self.conv4_1 = nn.Conv2d(2 * nFeats + nOutChannels_1, nFeats, 1, 1, 0, bias=False)
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
                ll_ = torch.cat([inter, ll], dim=1)
                try:
                    ll_ = torch.mul(ll_, torch.argmax(tmpOut[:, :, :, :], dim=1).view([tmpOut.shape[0], 1, tmpOut.shape[2], tmpOut.shape[3]]).float())
                except RuntimeError:
                    print('sdfe')
                inter = self.conv4_0(ll_)
            elif i == 1:
                tmpOut = self.conv2_1(ll)
                out.insert(i, tmpOut)
                ll_ = torch.cat([inter, ll, tmpOut], dim=1)
                inter = self.conv4_1(ll_)
            elif i == 2:
                tmpOut = self.conv2_2(ll)
                out.insert(i, tmpOut)
        return out


def main():
    if mode == 'train':
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        model = creatModel()
        model.cuda()
        loss1_background = nn.CrossEntropyLoss().cuda()
        loss2_skeleton = nn.CrossEntropyLoss().cuda()
        loss3_keypoints = nn.MSELoss().cuda()
        loss_epoch_array = []
        loss1_epoch_array = []
        loss2_epoch_array = []
        loss3_epoch_array = []
        step_epoch_array = []
        accuracy_array = []
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        imgLoader_train_coco = data.DataLoader(
            myImageDataset_COCO(train_set_coco, train_image_dir_coco, transform=mytransform), batch_size=batch_size,
            shuffle=True, num_workers=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        model, opt = amp.initialize(model, opt, opt_level="O1")
        model.train()
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10)
        # imgLoader_eval_coco = data.DataLoader(
        #     myImageDataset_COCO(eval_set_coco, eval_image_dir_coco, transform=mytransform), batch_size=batch_size,
        #     shuffle=True, num_workers=4)

        if retrain or not os.path.isfile(save_model_name):
            epoch = 0
        else:
            state = torch.load(save_model_name)
            model.load_state_dict(state['state_dict'])
            opt.load_state_dict(state['optimizer'])
            epoch = state['epoch']
            loss_epoch_array = state['loss']
            loss1_epoch_array = state['loss1']
            loss2_epoch_array = state['loss2']
            loss3_epoch_array = state['loss3']
            step_epoch_array = state['step']

        while epoch <= epochs:
            loss_record_array = []
            loss1_record_array = []
            loss2_record_array = []
            loss3_record_array = []
            step_array = []
            for i, [x_, y_keypoints, y_skeleton, y_background] in enumerate(imgLoader_train_coco, 0):
                bx_, by_keypoints, by_skeleton, by_background = x_.cuda(), y_keypoints.cuda(), y_skeleton.cuda(), y_background.cuda()
                result = model(bx_)
                loss_1 = loss1_background.forward(result[0], by_background)
                loss_2 = loss2_skeleton.forward(result[1], by_skeleton)
                loss_3 = loss3_keypoints.forward(result[2], by_keypoints)
                losses = loss_1 + loss_2 + loss_3
                opt.zero_grad()
                with amp.scale_loss(losses, opt) as scaled_loss:
                    scaled_loss.backward()
                # losses.backward()
                opt.step()
                # scheduler.step(losses)
                if i % 50 == 0:
                    loss_record = losses.cpu().data.numpy()
                    loss1_record = loss_1.cpu().data.numpy()
                    loss2_record = loss_2.cpu().data.numpy()
                    loss3_record = loss_3.cpu().data.numpy()
                    print('[{}/{}][{}/{}] Loss: {} Loss_1: {} Loss_2: {} Loss_3: {}'.format(
                        epoch, epochs, i, len(imgLoader_train_coco), loss_record,
                        loss1_record, loss2_record, loss3_record
                    ))
                    steps = i + len(imgLoader_train_coco) * epoch
                    loss_record_array.append(loss_record)
                    loss1_record_array.append(loss1_record)
                    loss2_record_array.append(loss2_record)
                    loss3_record_array.append(loss3_record)
                    step_array.append(steps)

            loss_epoch_array.append(loss_record_array)
            loss1_epoch_array.append(loss1_record_array)
            loss2_epoch_array.append(loss2_record_array)
            loss3_epoch_array.append(loss3_record_array)
            step_epoch_array.append(step_array)
            # accuracy_array.append(accuracy)
            plt.plot(np.reshape(step_epoch_array, -1), np.reshape(loss_epoch_array, -1),
                     np.reshape(step_epoch_array, -1), np.reshape(loss1_epoch_array, -1),
                     np.reshape(step_epoch_array, -1), np.reshape(loss2_epoch_array, -1),
                     np.reshape(step_epoch_array, -1), np.reshape(loss3_epoch_array, -1))
            plt.savefig(loss_img)
            # accuracy_array.append(accuracy)
            epoch += 1
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'loss': loss_epoch_array,
                'loss1': loss1_epoch_array,
                'loss2': loss2_epoch_array,
                'loss3': loss3_epoch_array,
                'step': step_epoch_array
            }
            torch.save(state, save_model_name)

    elif mode == 'test':
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        model = creatModel()
        model.cuda().half()
        state = torch.load(save_model_name)
        model.load_state_dict(state['state_dict'])
        epoch = state['epoch']
        loss_array = state['loss']
        test_mode = 'test'
        if test_mode == 'coco':

            imgLoader_val_coco = data.DataLoader(
                myImageDataset_COCO(train_set_coco, train_image_dir_coco, transform=mytransform), batch_size=1,
                shuffle=True, num_workers=1)
            for i, [val_image, val_keypoints, val_skeleton] in enumerate(imgLoader_val_coco):
                bx_ = val_image.cuda().half()
                result = model.forward(bx_)
                # accuracy = pckh(result[3], label.cuda().half())
                # print(accuracy)
                results = result[3].cpu().float().data.numpy()
                # image = (image.cpu().float().numpy()[0].transpose((1, 2, 0)) * 255).astype('uint8')
                # image = Image.fromarray(image)
                image = (val_image[0] + 1) / 2
                image = transforms.ToPILImage()(image)
                draw = ImageDraw.Draw(image)
                for i in range(37):
                    plt.subplot(3, 19, i + 1)
                    result = results[0, i, :, :]

                    plt.imshow(result)
                    # for i in range(38):
                    #     x = result[0, i, :, :]
                    #     ys, xs = np.multiply(np.where(x == np.max(x)), 4)
                    width = 5
                    draw.ellipse([xs - width, ys - width, xs + width, ys + width], fill=(0, 255, 0),
                                 outline=(255, 0, 0))

                del draw
                plt.subplot(3, 1, 3)
                plt.imshow(image)
                plt.show()

        elif test_mode == 'test':
            image = Image.open('test_img/im1.jpg').resize([256, 256])
            image_normalize = (mytransform(image)).unsqueeze(0).cuda().half()
            result = model.forward(image_normalize)
            # accuracy = pckh(result[3], label.cuda().half())
            # print(accuracy)
            results = result[0].cpu().float().data.numpy()
            # image = (image.cpu().float().numpy()[0].transpose((1, 2, 0)) * 255).astype('uint8')
            # image = Image.fromarray(image)
            plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)  # 调整子图间距
            draw = ImageDraw.Draw(image)
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.imshow(results[0, 1, :, :])
            plt.show()
            plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)
            results = result[1].cpu().float().data.numpy()
            for i in range(nOutChannels_1):
                plt.subplot(3, int(nOutChannels_1 / 2), i + 1)
                result_print = results[0, i, :, :]
                plt.imshow(result_print)
            plt.subplot(3, 1, 3)
            plt.imshow(image)
            plt.show()
            plt.subplots_adjust(wspace=0.1, hspace=0, left=0.03, bottom=0.03, right=0.97, top=1)
            results = result[2].cpu().float().data.numpy()
            for i in range(17):
                plt.subplot(3, 9, i + 1)
                result_print = results[0, i, :, :]

                # y_point, x_point = np.multiply(np.where(result_print == np.max(result)), 4)
                plt.imshow(result_print)
                # width = 2
                # draw.ellipse([x_point - width, y_point - width, x_point + width, y_point + width], fill=(int(255/17*i), int(255/17*i), int(255/17*i)), outline=(int(255/17*i), int(255/17*i), int(255/17*i)))

            del draw
            plt.subplot(3, 1, 3)
            plt.imshow(image)
            plt.show()

        print('yyy')


if __name__ == '__main__':
    main()
