import numpy as np
import matplotlib.pyplot as plt
from utilities3 import *
import random
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

j = 0     # 第 j 组数据

path = 'pred/pred_vesicle/'
dataloader = MatReader(path + 'ns_fourier_3d_rnn_V500_T10_N400_ep150_m12_w64.mat')
u_data = dataloader.read_field('test_u')[:, :, :, :, :].reshape(1092, 3, 3, 60)
predx = dataloader.read_field('pred')[:, :, :, :, :].reshape(1092, 3, 3, 60)

u_data = u_data.permute(3, 0, 1, 2)
u_data = u_data.permute(0, 2, 1, 3)
predx = predx.permute(3, 0, 1, 2)
predx = predx.permute(0, 2, 1, 3)

scipy.io.savemat('pred/pred_vesicle/pred_test' + '.mat', mdict={'u_data': u_data.cpu().numpy(), 'predx': predx.cpu().numpy()})

ux = u_data[:, :, :, :, 0]
uy = u_data[:, :, :, :, 1]
uz = u_data[:, :, :, :, 2]
ux = ux.permute(3, 0, 1, 2)
uy = uy.permute(3, 0, 1, 2)
uz = uz.permute(3, 0, 1, 2)
ux = np.reshape(ux, (60, -1))
uy = np.reshape(uy, (60, -1))
uz = np.reshape(uz, (60, -1))

uxp = predx[:, :, :, :, 0]
uyp = predx[:, :, :, :, 1]
uzp = predx[:, :, :, :, 2]
uxp = uxp.permute(3, 0, 1, 2)
uyp = uyp.permute(3, 0, 1, 2)
uzp = uzp.permute(3, 0, 1, 2)
uxp = np.reshape(uxp, (60, -1))
uyp = np.reshape(uyp, (60, -1))
uzp = np.reshape(uzp, (60, -1))

size_x, size_y, size_z = 64, 64, 64
gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(size_x, 1, 1).repeat([ 1, size_y, size_z])
gridx = np.reshape(gridx, (1, -1), order='C')
gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, size_y, 1).repeat([ size_x, 1, size_z])
gridy = np.reshape(gridy, (1, -1), order='C')
gridz = torch.tensor(np.linspace(-1, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, size_z).repeat([ size_x, size_y, 1])
gridz = np.reshape(gridz, (1, -1), order='C')


# for i in range(0, 10, 4):
#     predx1 = predx[:, :, :, i]
#     predx1 = np.reshape(predx1, (1, -1), order='C')
#     u1 = u_data[:, :, :, i]
#     u1 = np.reshape(u1, (1, -1), order='C')
#
#     fig = plt.figure(figsize=(6.4, 4.8), dpi=72)
#
#     # 3D散布図
#     ax = Axes3D(fig)
#     sc = ax.scatter(gridx[0: 64*64*32], gridy[0: 64*64*32], gridz[0: 64*64*32], c=u1[0: 64*64*32], cmap=cm.jet)
#
#     ax.set_title("Title", fontsize=8)
#     ax.set_xlabel("X", fontsize=7)
#     ax.set_ylabel("Y", fontsize=7)
#     ax.set_zlabel("Z", fontsize=7)
#     # tick labelのフォントサイズ
#     plt.tick_params(labelsize=6)
#
#     cb = fig.colorbar(sc, shrink=0.6)
#     # 水平にするときはorientationにhorizontalを追加
#     # cb=fig.colorbar(sc,shrink = 0.6, orientation="horizontal")
#
#     # カラーバーのlabel、tick label
#     cb.set_label("Color", fontsize=7)
#     cb.ax.tick_params(labelsize=6)
#
#     plt.show()

scipy.io.savemat('pred/pred10/test/pred_test' + '.mat', mdict={'gridx': gridx.cpu().numpy(), 'gridy': gridy.cpu().numpy(),
                                                   'gridz': gridz.cpu().numpy(), 'ux': ux.cpu().numpy(), 'uy': uy.cpu().numpy(),
                                                   'uz': uz.cpu().numpy(), 'uxp': uxp.cpu().numpy(), 'uyp': uyp.cpu().numpy(),
                                                   'uzp': uzp.cpu().numpy()})


# uxt3 = ux[:,:,:,3]
# uxt4 = ux[:,:,:,4]
# uxt5 = ux[:,:,:,5]
# print(uxt0[0,0,0], uxt1[0,0,0], uxt2[0,0,0], uxt3[0,0,0], uxt4[0,0,0], uxt5[0,0,0])