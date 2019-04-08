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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nModules = 2
nFeats = 256
nStack = 4
nKeypoint = 17
nSkeleton = 19
nOutChannels = nKeypoint + nSkeleton + 2
epochs = 50
batch_size = 16
keypoints = 17
skeleton = 20

mode = 'test'
save_model_name = 'params_1_coco_skeleton_and_keypoints.pkl'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
# eval_set_coco = '/data/COCO/COCO2014/annotations/person_keypoints_val2014.json'
train_image_dir_coco = '/data/COCO/COCO2017/train2017/'
# eval_image_dir_coco = '/data/COCO/COCO2014/val2014'

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
        return image, torch.Tensor(np.array(Label_map_keypoints)).long(), torch.Tensor(
            np.array(Label_map_skeleton)).long()


class ResidualBlock(nn.Module):
    def __init__(self, numIn, numOut):
        super(ResidualBlock, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn1 = nn.BatchNorm2d(numIn)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(numIn, int(numOut / 2), 1, 1)
        self.bn2 = nn.BatchNorm2d(int(numOut / 2))
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(int(numOut / 2), int(numOut / 2), 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(int(numOut / 2))
        self.relu = nn.ReLU(True)
        self.conv3 = nn.Conv2d(int(numOut / 2), numOut, 1, 1)
        self.conv4 = nn.Conv2d(numIn, numOut, 1, 1)

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
        out = self.conv3(x)
        if self.numIn != self.numOut:
            residual = self.conv4(residual)
        out += residual
        return out


class hourglass(nn.Module):
    def __init__(self, n, f):
        super(hourglass, self).__init__()
        self.n = n
        self.f = f
        self.residual_block = ResidualBlock(f, f)
        if n > 1:
            self.hourglass1 = hourglass(n - 1, f)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        up1 = x
        for i in range(nModules):
            up1 = self.residual_block(up1)
        low1 = self.maxpool(x)
        for i in range(nModules):
            low1 = self.residual_block(low1)
        if self.n > 1:
            low2 = self.hourglass1(low1)
        else:
            low2 = low1
            for i in range(nModules):
                low2 = self.residual_block(low2)
        low3 = low2
        for i in range(nModules):
            low3 = self.residual_block(low3)
        up2 = nn.functional.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)
        out = up1 + up2
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
        self.residual1 = ResidualBlock(64, 128)
        self.max_pool1 = nn.MaxPool2d(2)
        self.residual2 = ResidualBlock(128, 128)
        self.residual3 = ResidualBlock(128, nFeats)
        self.hourglass1 = hourglass(4, nFeats)
        self.residual4 = ResidualBlock(nFeats, nFeats)
        self.lin = lin(nFeats, nFeats)
        self.conv2 = nn.Conv2d(nFeats, nOutChannels, 1, 1, 0)
        self.conv3 = nn.Conv2d(nFeats, nFeats, 1, 1, 0)
        self.conv4 = nn.Conv2d(nOutChannels, nFeats, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.max_pool1(x)
        x = self.residual2(x)
        x = self.residual3(x)

        out = []
        inter = x
        for i in range(nStack):
            hg = self.hourglass1(inter)
            ll = hg
            for j in range(nModules):
                ll = self.residual4(ll)
            ll = self.lin(ll)
            tmpOut = self.conv2(ll)
            out.insert(i, tmpOut)

            if i < nStack:
                ll_ = self.conv3(ll)
                tmpOut[:, 19:, :, :] = (tmpOut[:, 19:, :, :].permute(1, 0, 2, 3) - tmpOut[:, 0, :, :]).permute(1, 0, 2,
                3) + tmpOut[:, sks, :, :][:, :, 0,:, :] + tmpOut[:, sks, :, :][:, :, 1, :, :]
                tmpOut_ = self.conv4(tmpOut)
                inter = ll_ + tmpOut_
        return out


# class creatModelD(nn.Module):
#     def __init__(self):
#         super(creatModelD, self).__init__()
#         self.conv1 = nn.Conv2d(nOutChannels + 3, 64, 3, 1, 1)
#         self.relu = nn.ReLU()
#         self.residual1 = ResidualBlock(64, 128)
#         self.residual2 = ResidualBlock(128, 128)
#         self.residual3 = ResidualBlock(128, nFeats)
#         self.hourglass = hourglass(4, nFeats)
#         self.residual4 = ResidualBlock(nFeats, nFeats)
#         self.lin = lin(nFeats, nFeats)
#         self.conv2 = nn.Conv2d(nFeats, nOutChannels, 1, 1, 0)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.residual1(x)
#         x = self.residual2(x)
#         x = self.residual3(x)
#         x = self.hourglass(x)
#         ll = x
#         for i in range(nModules):
#             ll = self.residual4(ll)
#         ll = lin(ll)
#         out = self.conv2(ll)
#         return out


# class PCKh(nn.Module):
#     def __init__(self):
#         super(PCKh, self).__init__()
#
#     def forward(self, x, target):
#         correct = 0
#         total = 0
#         for i in range(batch_size):
#             head_heat_map = target[i, 13, :, :]
#             head_ys = torch.max(torch.max(head_heat_map, 1)[0], 0)[1]
#             head_xs = torch.max(head_heat_map, 1)[1][head_ys]
#             neck_heat_map = target[i, 1, :, :]
#             neck_ys = torch.max(torch.max(neck_heat_map, 1)[0], 0)[1]
#             neck_xs = torch.max(neck_heat_map, 1)[1][neck_ys]
#             standard = torch.sqrt((torch.pow(head_ys - neck_ys, 2) + torch.pow(head_xs - neck_xs, 2)).float()) / 2
#             for j in range(14):
#                 label_heat_map = target[i, j, :, :]
#                 if torch.max(label_heat_map) == 0:
#                     continue
#                 label_ys = torch.max(torch.max(label_heat_map, 1)[0], 0)[1]
#                 label_xs = torch.max(label_heat_map, 1)[1][head_ys]
#                 predict_heat_map = x[i, j, :, :]
#                 predict_ys = torch.max(torch.max(predict_heat_map, 1)[0], 0)[1]
#                 predict_xs = torch.max(label_heat_map, 1)[1][head_ys]
#                 if torch.sqrt(
#                         (torch.pow(label_ys - predict_ys, 2) + torch.pow(label_xs - predict_xs, 2)).float()) < standard:
#                     correct += 1
#                 total += 1
#         return correct / total


def main():
    # loss1 = nn.MSELoss().cuda().half()
    # loss2 = nn.MSELoss().cuda().half()
    # loss3 = nn.MSELoss().cuda().half()
    # loss4 = nn.MSELoss().cuda().half()

    model = creatModel()
    model.cuda()
    # pckh = PCKh()
    # dataset = myImageDataset(rootdir, jointsdir)
    # x_, y_ = dataset.__getitem__(0)

    # imgLoader_train = data.DataLoader(myImageDataset(train_set, jointsdir, transform=mytransform), batch_size=batch_size,
    #                             shuffle=True,
    #                             num_workers=8)
    # imgLoader_eval = data.DataLoader(myImageDataset(eval_set, jointsdir, transform=mytransform),
    #                                   batch_size=batch_size,
    #                                   shuffle=True,
    #                                   num_workers=8)

    # test = myImageDataset_COCO(train_set_coco, train_image_dir_coco)
    # x, y = test.__getitem__(0)
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if mode == 'train':
        loss1_keypoints = nn.CrossEntropyLoss().cuda()
        loss1_skeleton = nn.CrossEntropyLoss().cuda()
        loss2_keypoints = nn.CrossEntropyLoss().cuda()
        loss2_skeleton = nn.CrossEntropyLoss().cuda()
        loss3_keypoints = nn.CrossEntropyLoss().cuda()
        loss3_skeleton = nn.CrossEntropyLoss().cuda()
        loss4_keypoints = nn.CrossEntropyLoss().cuda()
        loss4_skeleton = nn.CrossEntropyLoss().cuda()
        loss_array = []
        accuracy_array = []
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        imgLoader_train_coco = data.DataLoader(
            myImageDataset_COCO(train_set_coco, train_image_dir_coco, transform=mytransform), batch_size=batch_size,
            shuffle=True, num_workers=20)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
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
            loss_array = state['loss']
        while epoch <= epochs:
            for i, [x_, y_keypoints, y_skeleton] in enumerate(imgLoader_train_coco, 0):
                bx_, by_keypoints, by_skeleton = x_.cuda(), y_keypoints.cuda(), y_skeleton.cuda()
                result = model(bx_)
                loss_1 = loss1_keypoints.forward(result[0][:, :18, :, :],
                                                 by_keypoints) + loss1_skeleton.forward(
                    result[0][:, 18:, :, :], by_skeleton)
                loss_2 = loss2_keypoints.forward(result[1][:, :18, :, :],
                                                 by_keypoints) + loss2_skeleton.forward(
                    result[1][:, 18:, :, :], by_skeleton)
                loss_3 = loss3_keypoints.forward(result[2][:, :18, :, :],
                                                 by_keypoints) + loss3_skeleton.forward(
                    result[2][:, 18:, :, :], by_skeleton)
                loss_4 = loss4_keypoints.forward(result[3][:, :18, :, :],
                                                 by_keypoints) + loss4_skeleton.forward(
                    result[3][:, 18:, :, :], by_skeleton)
                losses = loss_1 + loss_2 + loss_3 + loss_4
                if math.isnan(losses.cpu().data.numpy()):
                    print('sde')

                opt.zero_grad()
                losses.backward()
                opt.step()
            with torch.no_grad():
                # dataiter = iter(imgLoader_eval_coco)
                # x_, y = dataiter.next()
                # bx_, by = x_.cuda().half(), y.cuda().half()
                # result = model(bx_)
                # accuracy = pckh(result[3], by)
                print(str(epoch) + ' ' + str(losses.cpu().data.numpy()) + ' ' + str(
                    loss_1.cpu().data.numpy()) + ' ' + str(
                    loss_2.cpu().data.numpy()) + ' ' + str(
                    loss_3.cpu().data.numpy()) + ' ' + str(
                    loss_4.cpu().data.numpy()))
                loss_array.append(loss_4.cpu().data.numpy())
                # accuracy_array.append(accuracy)
                x = np.linspace(0, epoch, epoch + 1)
                plt.plot(x, loss_array)
                plt.savefig(loss_img)
                # accuracy_array.append(accuracy)
                epoch += 1
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'loss': loss_array
                }
                torch.save(state, save_model_name)

    elif mode == 'test':
        state = torch.load(save_model_name)
        model.load_state_dict(state['state_dict'])
        epoch = state['epoch']
        loss_array = state['loss']
        image = Image.open('test_img/images_3.jpeg').resize([256, 256])
        image_normalize = (mytransform(image)).unsqueeze(0).cuda()
        result = model.forward(image_normalize)
        # accuracy = pckh(result[3], label.cuda().half())
        # print(accuracy)
        result = result[3].cpu().data.numpy()
        # image = (image.cpu().float().numpy()[0].transpose((1, 2, 0)) * 255).astype('uint8')
        # image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for i in range(38):
            plt.subplot(3, 19, i + 1)
            plt.imshow(result[0, i, :, :])
        # for i in range(38):
        #     x = result[0, i, :, :]
        #     ys, xs = np.multiply(np.where(x == np.max(x)), 4)
        #     width = 5
        #     draw.ellipse([xs - width, ys - width, xs + width, ys + width], fill=(0, 255, 0), outline=(255, 0, 0))

        del draw
        plt.subplot(3, 1, 3)
        plt.imshow(image)
        plt.show()

        for i in range(38):
            plt.subplot(3, 19, i + 1)
            plt.imshow(result[0, i, :, :])

        plt.subplot(3, 1, 3)
        plt.imshow(image)
        plt.show()

        print('yyy')


if __name__ == '__main__':
    main()
