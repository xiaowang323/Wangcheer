import numpy as np
import matplotlib.pyplot as plt
from utilities3 import *
import random

path = 'data/vesicle/'
dataloader = MatReader(path + '64_2.mat')
dataloader2 = MatReader(path + 'wrinkshtest.mat')
ct = dataloader2.read_field('ct')[:,:]

u1 = dataloader.read_field('x')[:,:]
u2 = dataloader.read_field('y')[:,:]
u3 = dataloader.read_field('z')[:,:]
j = 301
uxx = u1[:, j:j+200:10]
uxx = uxx.reshape(1, -1, 20)
ux = uxx
uyy = u2[:, j:j+200:10]
uyy = uyy.reshape(1, -1, 20)
uy = uyy
uzz = u3[:, j:j+200:10]
uzz = uzz.reshape(1, -1, 20)
uz = uzz

for j in range(302, 601, 1):
    uxx = u1[:, j:j+200:10]
    uxx = uxx.reshape(1, -1, 20)
    ux = torch.cat((ux, uxx), dim=0)
    uyy = u2[:, j:j + 200:10]
    uyy = uyy.reshape(1, -1, 20)
    uy = torch.cat((uy, uyy), dim=0)
    uzz = u3[:, j:j+200:10]
    uzz = uzz.reshape(1, -1, 20)
    uz = torch.cat((uz, uzz), dim=0)

reader1 = torch.stack((ux, uy, uz), dim=3)

##########
path = 'data/vesicle/test/'
dataloader = MatReader(path + '64_2.mat')
u1 = dataloader.read_field('x')[:,:]
u2 = dataloader.read_field('y')[:,:]
u3 = dataloader.read_field('z')[:,:]
j = 301
uxx = u1[:, j:j+200:10]
uxx = uxx.reshape(1, -1, 20)
ux = uxx
uyy = u2[:, j:j+200:10]
uyy = uyy.reshape(1, -1, 20)
uy = uyy
uzz = u3[:, j:j+200:10]
uzz = uzz.reshape(1, -1, 20)
uz = uzz

for j in range(302, 601, 1):
    uxx = u1[:, j:j+200:10]
    uxx = uxx.reshape(1, -1, 20)
    ux = torch.cat((ux, uxx), dim=0)
    uyy = u2[:, j:j + 200:10]
    uyy = uyy.reshape(1, -1, 20)
    uy = torch.cat((uy, uyy), dim=0)
    uzz = u3[:, j:j+200:10]
    uzz = uzz.reshape(1, -1, 20)
    uz = torch.cat((uz, uzz), dim=0)

reader2 = torch.stack((ux, uy, uz), dim=3)

reader = torch.cat((reader1, reader2), dim=0)
data_num = reader.shape[0] #得到样本数
index = np.arange(data_num) #生成下标
np.random.shuffle(index)
reader = reader[index]
scipy.io.savemat(path+'reader_vesicle.mat', mdict={'reader': reader.cpu().numpy()})
