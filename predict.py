# coding: utf-8
import cv2
import pickle
from network import network

# データ
Img = cv2.imread('9.png', cv2.IMREAD_GRAYSCALE)
Img = Img.astype('float32')
Img /= 255.0
Img = Img.reshape([1,1,28,28])

# hparams
with open('hparams_988.pkl', 'rb') as f:
    hparams = pickle.load(f)
n = hparams['n']
lr = hparams['lr']
conv_pool_params = hparams['conv_pool_params']
dropout_ratio = hparams['dropout_ratio']
weight_V = hparams['weight_V']
weight_H = hparams['weight_H']
minibatch_size = hparams['minibatch_size']

# ネットワーク
nw = network(n, lr, conv_pool_params, 
             dropout_ratio, 
             weight_V, weight_H, 
             minibatch_size)

# params
with open('params_988.pkl', 'rb') as f:
    params = pickle.load(f)
for i in range(1, 4):
    nw.layers['conv0_'+str(i)].W = params['conv0_'+str(i)+'_W']
    nw.layers['conv0_'+str(i)].b = params['conv0_'+str(i)+'_b']
    nw.layers['conv1_'+str(i)].W = params['conv1_'+str(i)+'_W']
    nw.layers['conv1_'+str(i)].b = params['conv1_'+str(i)+'_b']
nw.layers['affine_4'].W = params['affine_4_W']
nw.layers['affine_4'].b = params['affine_4_b']
nw.layers['affine_5'].W = params['affine_5_W']
nw.layers['affine_5'].b = params['affine_5_b']

# フラグ
for layer in nw.layers_with_flgs:
    layer.trained_flg = 1

# 推論
nw.predict(Img)