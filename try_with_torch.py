import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import scipy.io
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

nModules = 2
nFeats = 256
nStack = 4
nOutChannels = 14

batch_size = 16


class myImageDataset(data.Dataset):
    def __init__(self, filepath, matdir, transform=None, dim=(256, 256), n_channels=3,
                 n_joints=14):
        'Initialization'
        self.mat = scipy.io.loadmat(matdir)
        self.dim = dim
        self.filepath = filepath
        self.list = os.listdir(filepath)
        self.n_channels = n_channels
        self.n_joints = n_joints
        self.transform = transform

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        image = Image.open(self.filepath + self.list[index]).convert('RGB')
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

            Gauss_map[k, :, :] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-temp)

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
    rootdir = '/data/lsp_dataset/images/'
    jointsdir = '/data/lsp_dataset/joints.mat'
    dataset = myImageDataset(rootdir, jointsdir)
    # x_, y_ = dataset.__getitem__(0)
    mytransform = transforms.Compose([
        transforms.ToTensor()
    ]
    )
    imgLoader = data.DataLoader(myImageDataset(rootdir, jointsdir, transform=mytransform), batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    mode = 'test'
    if mode == 'train':
        epochs = 1000
        for epoch in range(epochs):
            for i, [x_, y] in enumerate(imgLoader, 0):
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
                accuracy = pckh(result[3], by)
            print(str(epoch) + ' ' + str(accuracy) + str(losses.data) + str(loss_1.data) + str(loss_2.data) + str(loss_3.data) + str(
                loss_4.data))
            torch.save(model.state_dict(), 'params.pkl')
    elif mode == 'test':
        model.load_state_dict(torch.load('params.pkl'))
        dataiter = iter(imgLoader)
        image, label = dataiter.next()
        result = model.forward(image.cuda())
        accuracy = pckh(result[3], label.cuda())
        print(accuracy)
        result = result[3].cpu().data.numpy()
        image = (image.cpu().numpy()[0].transpose((1, 2, 0)) * 255).astype('uint8')
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for i in range(14):
            x = result[0, i, :, :]
            ys, xs = np.multiply(np.where(x == np.max(x)), 4)
            width = 5
            draw.ellipse([xs - width, ys - width, xs + width, ys + width], fill=(0, 255, 0), outline=(255, 0, 0))

        del draw
        plt.imshow(image)
        plt.show()

        print('yyy')


if __name__ == '__main__':
    main()
