# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# plot3d
# 損失関数が凸関数か確認
# インライン表示を無効: ツール>ipythonコンソール>グラフィックス>バックエンド>無効
# 設定変更を反映するにはspyderを再起動
x = np.arange(0.1, 1.0, 0.1)
y = np.arange(0.1, 1.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = X*np.log(Y) + (1-X)*np.log(1-Y)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot_wireframe(X, Y, Z)
#ax.plot_surface(X, Y, Z)
plt.show()


# plot_graphs with subplots
t = np.linspace(-np.pi, np.pi, 1000)

x1 = np.sin(2*t)
x2 = np.cos(2*t)
x3 = x1 + x2

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))

axes[0,0].plot(t, x1)
axes[0,0].set_title('sin')
axes[0,0].set_xlabel('t')
axes[0,0].set_ylabel('x')
axes[0,0].set_xlim(-np.pi, np.pi)
axes[0,0].grid(True)

axes[0,1].plot(t, x2, linewidth=2)
axes[0,1].set_title('cos')
axes[0,1].set_xlabel('t')
axes[0,1].set_ylabel('x')
axes[0,1].set_xlim(-np.pi, np.pi)
axes[0,1].grid(True)

axes[1,0].plot(t, x3, linewidth=2)
axes[1,0].set_title('sin+cos')
axes[1,0].set_xlabel('t')
axes[1,0].set_ylabel('x')
axes[1,0].set_xlim(-np.pi, np.pi)
axes[1,0].grid(True)

axes[1,1].axis('off')

fig.show()


# plot_graphs with gridspec
t = np.linspace(-np.pi, np.pi, 1000)

x1 = np.sin(2*t)
x2 = np.cos(2*t)
x3 = x1 + x2

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,2)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[1,:])

ax1.plot(t, x1, linewidth=2)
ax1.set_title('sin')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_xlim(-np.pi, np.pi)
ax1.grid(True)

ax2.plot(t, x2, linewidth=2)
ax2.set_title('cos')
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_xlim(-np.pi, np.pi)
ax2.grid(True)

ax3.plot(t, x3, linewidth=2)
ax3.set_title('sin+cos')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_xlim(-np.pi, np.pi)
ax3.grid(True)

fig.show()