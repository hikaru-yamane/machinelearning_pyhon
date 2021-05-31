# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from layers import *
from optimizers import *

class network:
    def __init__(self, n0, n1, n2, minibatch_size, 
                 lr, weight_V, weight_H, dropout_ratio):
        self.layers = OrderedDict()
        self.layers['affine1'] = affine(n0, n1)
#        self.layers['BN1'] = batch_normalization(n1)
        self.layers['actfunc1'] = relu()
#        self.layers['dropout1'] = dropout(dropout_ratio)
        self.layers['affine2'] = affine(n1, n2)
        self.layers['lastlayer'] = softmax_with_loss(minibatch_size)
        # パラメータのあるlayerのみ
        # ２つのfor文を１つで書けるが可読性のためやらない
        self.layers_with_params = [] # リストの作成
        for key, layer in self.layers.items():
            if 'affine' in key: self.layers_with_params.append(layer)
            if 'BN' in key: self.layers_with_params.append(layer)
        for layer in self.layers_with_params:
            layer.W_optimizer = gradient_descent(lr)
            layer.b_optimizer = gradient_descent(lr)
    
    def get_gradients(self, X):
        # FP
        for layer in self.layers.values():
            X = layer.forward(X)
        # BP
        layers_list = list(self.layers.values())
        layers_list.reverse()
        dX = 1 # dloss
        for layer in layers_list:
            dX = layer.backward(dX)
    
    def get_accuracy(self, X, label):
        # lastlayerを除いたlayer
        m = X.shape[0]
        layers_list = list(self.layers.values()) # 順序付き辞書にはインデックスがない
        del layers_list[-1] # 最後の要素を削除
        for layer in layers_list:
            X = layer.forward(X)
        max_ind = np.argmax(X, axis=1).reshape([-1,1])
        acc = np.sum( max_ind==label ) / m * 100
        
        return acc
    
    def update_parameters(self):
        # パラメータのあるlayerのみ
        for layer in self.layers_with_params:
            layer.W = layer.W_optimizer.update(layer.W, layer.dW)
            layer.b = layer.b_optimizer.update(layer.b, layer.db)
    
    def predict(self, X):
        # 画像描画
        X = X.reshape([28,28])
        plt.imshow(X)
        plt.show()
        # 推論
        # lastlayerを除いたlayer
        X = X.reshape([1,-1])
        layers_list = list(self.layers.values())
        del layers_list[-1] # 最後の要素を削除
        for layer in layers_list:
            X = layer.forward(X)
        max_ind = np.argmax(X, axis=1).reshape([-1,1])
        print('predicted number: {}'.format(max_ind))