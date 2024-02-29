"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

from fourier_3d_time_model import *
# from fourier_3d_time_model import SpectralConv3d_fast


torch.manual_seed(0)
np.random.seed(0)

# ################################################################
# # fourier layer
# ################################################################
#
# class SpectralConv3d_fast(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
#         super(SpectralConv3d_fast, self).__init__()
#
#         """
#         3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
#         """
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2
#         self.modes3 = modes3
#
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#         self.weights3 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#         self.weights4 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#
#         # Complex multiplication
#
#     def compl_mul3d(self, input, weights):
#         # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
#         return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
#
#     def forward(self, x):
#         batchsize = x.shape[0]
#         # Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
#
#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
#                              dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
#         out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
#         out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
#
#         # Return to physical space
#         x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
#         return x
#
# class FNO3d(nn.Module):
#     def __init__(self, modes1, modes2, modes3, width):
#         super(FNO3d, self).__init__()
#
#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
#
#         input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
#         input shape: (batchsize, x=64, y=64, c=12)
#         output: the solution of the next timestep
#         output shape: (batchsize, x=64, y=64, c=1)
#         """
#
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.modes3 = modes3
#         self.width = width
#         self.padding = 6  # pad the domain if input is non-periodic
#         self.fc0 = nn.Linear(33, self.width)
#         # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y, z), ..., u(t-1, x, y, z),  x, y, z)
#
#         self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.w0 = nn.Conv3d(self.width, self.width, 1)
#         self.w1 = nn.Conv3d(self.width, self.width, 1)
#         self.w2 = nn.Conv3d(self.width, self.width, 1)
#         self.w3 = nn.Conv3d(self.width, self.width, 1)
#         self.bn0 = torch.nn.BatchNorm3d(self.width)
#         self.bn1 = torch.nn.BatchNorm3d(self.width)
#         self.bn2 = torch.nn.BatchNorm3d(self.width)
#         self.bn3 = torch.nn.BatchNorm3d(self.width)
#
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 3)
#
#     def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#
#         x = x.permute(0, 4, 1, 2, 3)
#         x = F.pad(x, [0, self.padding])
#         # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
#
#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2
#         x = F.gelu(x)
#
#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)
#
#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)
#
#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2
#
#         # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
#         x = x[..., :-self.padding]
#         x = x.permute(0, 2, 3, 4, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x
#
#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
#         gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
#         gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
#         return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# configs
################################################################

# TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
path = 'data/vesicle10/test/'
# TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'

ntrain = 500
ntest = 100

modes = 12
width = 48

batch_size = 10
batch_size2 = batch_size

epochs = 200
learning_rate = 0.001
scheduler_step = 100   # ?
scheduler_gamma = 0.5  # ?

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

PATH = 'ns_fourier_3d_rnn_V500_T10_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '.pt'
path_model = 'model/'+PATH
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path

sub = 1  # 在数据集中间隔sub取作训练（测试）数据 如在128*128数据集中取64*64来训练和测试网络
# S = 64
T_in = 10  # 输入数据
T = 10  # 预测T个数据
step = 1

################################################################
# load data
################################################################

reader = MatReader(path + 'reader_vesicle_train.mat')
train_a = reader.read_field('reader5')[:ntrain, ::sub, :T_in, ::sub]
train_u = reader.read_field('reader5')[:ntrain, ::sub, T_in:T+T_in, ::sub]

test_a = reader.read_field('reader5')[-ntest:, ::sub, :T_in, ::sub]
test_u = reader.read_field('reader5')[-ntest:, ::sub, :T_in, ::sub]

print(train_u.shape)
print(test_u.shape)
S = train_u.shape[-3]
# assert (S == train_u.shape[-3])
assert (T == train_u.shape[-2])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################

model = FNO3d(modes, modes, modes, width).cuda()
# model = torch.load('model/fourier_3d_rnn_V500_T10_N400_ep50_m12_w48.pt')
# pred_path = 'fourier_3d_rnn_V500_T10_N400_ep50_m12_w48'

print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[:, :, :, :, t:t + step, :]
            y = y.reshape(batch_size, S, S, S, 3)
            xxx = xx.reshape(batch_size, S, S, S, 3*T_in)

            im = model(xxx)
            im = im.reshape(batch_size, S, S, S, 1, 3)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)

            xx = torch.cat((xx[..., step:, :], im), dim=-2)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[:, :, :, :, t:t + step, :]
                y = y.reshape(batch_size, S, S, S, 3)
                xxx = xx.reshape(batch_size, S, S, S, 3 * T_in)
                im = model(xxx)
                im = im.reshape(batch_size, S, S, S, 1, 3)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)

                xx = torch.cat((xx[..., step:, :], im), dim=-2)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)

torch.save(model, path_model)

# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# test_l2_step = 0
# test_l2_full = 0
# with torch.no_grad():
#     for x, y in test_loader:
#         for xx, yy in test_loader:
#             loss = 0
#             xx = xx.to(device)
#             yy = yy.to(device)
#
#             for t in range(0, T, step):
#                 y = yy[:, :, :, :, t:t + step, :]
#                 y = y.reshape(batch_size, S, S, S, 3)
#                 xxx = xx.reshape(batch_size, S, S, S, 3 * T_in)
#                 im = model(xxx)
#                 im = im.reshape(batch_size, S, S, S, 1, 3)
#                 loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
#
#                 if t == 0:
#                     pred = im
#                 else:
#                     pred = torch.cat((pred, im), -2)
#
#                 xx = torch.cat((xx[..., step:, :], im), dim=-2)
#
#             test_l2_step += loss.item()
#             test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
#             print(index, test_l2_step / ntest / (T / step), test_l2_full / ntest)
#             index = index + 1
#
# scipy.io.savemat('pred/'+pred_path+'.mat', mdict={'pred': pred.cpu().numpy()})