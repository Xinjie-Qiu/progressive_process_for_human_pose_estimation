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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nModules = 2
nFeats = 256
nStack = 4
nOutChannels = 17
epochs = 1000
batch_size = 16
keypoints = 17

mode = 'train'
save_model_name = 'params_2_coco.pkl'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO2014/annotations/person_keypoints_train2014.json'
eval_set_coco = '/data/COCO2014/annotations/person_keypoints_val2014.json'
train_image_dir_coco = '/data/COCO2014/train2014/COCO_train2014_'
eval_image_dir_coco = '/data/COCO2014/val2014/COCO_val2014_'

rootdir = '/data/lsp_dataset/images/'


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
            image_after_transform = self.transform(image)
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

                temp = 100 * ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

                Gauss_map[k, :, :] = np.exp(-temp)

        return image_after_transform, torch.Tensor(Gauss_map)


class myImageDataset(data.Dataset):
    def __init__(self, filename, matdir, transform=None, dim=(256, 256), n_channels=3,
                 n_joints=14):
        'Initialization'
        self.mat = scipy.io.loadmat(matdir)
        self.dim = dim
        file = open(filename)
        self.list = file.readlines()
        self.n_channels = n_channels
        self.n_joints = n_joints
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image = Image.open((rootdir + self.list[index]).strip()).convert('RGB')
        w, h = image.size
        image = image.resize([256, 256])
        if self.transform is not None:
            image = self.transform(image)

        number = int(self.list[index][2:6]) - 1
        Gauss_map = np.zeros([14, 64, 64])
        for k in range(14):
            xs = self.mat['joints'][0][k][number] / w * 64
            ys = self.mat['joints'][1][k][number] / h * 64
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
                tmpOut_ = self.conv4(tmpOut)
                inter = ll_ + tmpOut_
        return out


class creatModelD(nn.Module):
    def __init__(self):
        super(creatModelD, self).__init__()
        self.conv1 = nn.Conv2d(nOutChannels + 3, 64, 3, 1, 1)
        self.relu = nn.ReLU()
        self.residual1 = ResidualBlock(64, 128)
        self.residual2 = ResidualBlock(128, 128)
        self.residual3 = ResidualBlock(128, nFeats)
        self.hourglass = hourglass(4, nFeats)
        self.residual4 = ResidualBlock(nFeats, nFeats)
        self.lin = lin(nFeats, nFeats)
        self.conv2 = nn.Conv2d(nFeats, nOutChannels, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.hourglass(x)
        ll = x
        for i in range(nModules):
            ll = self.residual4(ll)
        ll = lin(ll)
        out = self.conv2(ll)
        return out


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


def main():
    loss1 = nn.MSELoss()
    loss2 = nn.MSELoss()
    loss3 = nn.MSELoss()
    loss4 = nn.MSELoss()
    model = creatModel()
    model.cuda()
    pckh = PCKh()
    pckh.cuda()
    jointsdir = '/data/lsp_dataset/joints.mat'
    # dataset = myImageDataset(rootdir, jointsdir)
    # x_, y_ = dataset.__getitem__(0)
    loss_array = []
    accuracy_array = []
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # imgLoader_train = data.DataLoader(myImageDataset(train_set, jointsdir, transform=mytransform), batch_size=batch_size,
    #                             shuffle=True,
    #                             num_workers=8)
    # imgLoader_eval = data.DataLoader(myImageDataset(eval_set, jointsdir, transform=mytransform),
    #                                   batch_size=batch_size,
    #                                   shuffle=True,
    #                                   num_workers=8)

    imgLoader_train_coco = data.DataLoader(myImageDataset_COCO(train_set_coco, train_image_dir_coco, transform=mytransform), batch_size=batch_size, shuffle=True, num_workers=1)
    imgLoader_eval_coco = data.DataLoader(myImageDataset_COCO(eval_set_coco, eval_image_dir_coco, transform=mytransform), batch_size=batch_size, shuffle=True, num_workers=1)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    if mode == 'train':
        for epoch in range(epochs):
            for i, [x_, y] in enumerate(imgLoader_train_coco, 0):
                bx_, by = x_.cuda(), y.cuda()
                result = model(bx_)
                loss_1 = loss1.forward(result[0], by)
                loss_2 = loss2.forward(result[1], by)
                loss_3 = loss3.forward(result[2], by)
                loss_4 = loss4.forward(result[3], by)
                losses = loss_1 + loss_2 + loss_3 + loss_4
                opt.zero_grad()
                losses.backward()
                opt.step()
            with torch.no_grad():
                dataiter = iter(imgLoader_eval_coco)
                x_, y = dataiter.next()
                bx_, by = x_.cuda(), y.cuda()
                result = model(bx_)
                accuracy = pckh(result[3], by)
                print(str(epoch) + ' ' + str(accuracy) + str(losses.data) + str(loss_1.data) + str(loss_2.data) + str(
                    loss_3.data) + str(
                    loss_4.data))
                loss_array.append(loss_4.cpu().data.numpy())
                accuracy_array.append(accuracy)
                torch.save(model.state_dict(), save_model_name)
        x = np.linspace(0, epochs-1, epochs)
        plt.plot(x, loss_array)
        plt.savefig('loss.png')
        plt.plot(x, accuracy_array)
        plt.savefig('accuracy.png')
        plt.show()
    elif mode == 'test':
        model.load_state_dict(torch.load(save_model_name))
        image = Image.open(rootdir + 'im0401.jpg').resize([256, 256])
        image_normalize = (mytransform(image)).unsqueeze(0)
        result = model.forward(image_normalize)
        # accuracy = pckh(result[3], label.cuda())
        # print(accuracy)
        result = result[3].cpu().data.numpy()
        # image = (image.cpu().numpy()[0].transpose((1, 2, 0)) * 255).astype('uint8')
        # image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for i in range(14):
            plt.subplot(3, 7, i + 1)
            plt.imshow(result[0, i, :, :])
        # for i in range(14):
        #     x = result[0, i, :, :]
        #     ys, xs = np.multiply(np.where(x == np.max(x)), 4)
        #     width = 5
        #     draw.ellipse([xs - width, ys - width, xs + width, ys + width], fill=(0, 255, 0), outline=(255, 0, 0))

        del draw
        plt.subplot(3, 1, 3)
        plt.imshow(image)
        plt.show()

        print('yyy')


if __name__ == '__main__':
    main()
