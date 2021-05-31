# coding: utf-8
import numpy as np
from layers import *

# 初期化
#Data = np.random.randn(2,2,2,2)
#Data = np.random.randn(2,8)
Data = np.array([[1],[2],[5],[1]])
#Data = np.random.rand(4,2)
#Data = np.arange(16).reshape([1,1,4,4])
#Data = np.random.randint(10,size=(1,1,4,4))
#layer = batch_normalization(1,3)
layer = batch_normalization_original(1)
#layer = batch_normalization_book(2,0)
#layer = convolution(2,2,2,2,1,2)
#layer.Label = np.zeros_like(Data) # lastlayer
#layer.Label[:,0] = 1 # lastlayer

# 解析微分
X = layer.forward(Data)
loss = np.sum(X**2) # 仮の損失関数
dX = 2*X
dX = layer.backward(dX)
#dX = layer.backward(1) # lastlayer

# 数値微分
epsilon = 1e-4
dX_num = np.zeros_like(Data)
it = np.nditer(Data, flags=['multi_index'])
while not it.finished:
    temp = Data[it.multi_index]
    Data[it.multi_index] = temp + epsilon
    X = layer.forward(Data)
#    X = Data.reshape([Data.shape[0],-1]) * layer.Mask # dropout
    loss_plus = np.sum(X**2)
#    loss_plus = layer.forward(Data) # lastlayer
    Data[it.multi_index] = temp - epsilon
    X = layer.forward(Data)
#    X = Data.reshape([Data.shape[0],-1]) * layer.Mask # dropout
    loss_minus = np.sum(X**2)
#    loss_minus = layer.forward(Data) # lastlayer
    dX_num[it.multi_index] = (loss_plus - loss_minus)/(2*epsilon)
    Data[it.multi_index] = temp
    it.iternext()

print((dX - dX_num).reshape([-1,1]))
print('|dX|: {:.5f}'.format(np.sum(np.absolute(dX))))