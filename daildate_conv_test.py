import torch
import numpy as np

if __name__ == '__main__':
    noise = torch.randn(1, 1, 10, 10)
    bx = torch.Tensor(noise)
    output = torch.nn.Conv2d(1, 1, 3, 1, 1, dilation=2)(bx)
    print('ete')
