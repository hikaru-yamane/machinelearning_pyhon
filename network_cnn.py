# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from layers import *
from optimizers import *

class network:
    def __init__(self, n, minibatch_size, 
                 conv_hparams, pool_hparams, 
                 dropout_ratio, 
                 opt_hparams):
        # ネットワーク
        self.layers = OrderedDict()
        self.layers['conv_1'] = convolution(conv_hparams['1']['mf'], conv_hparams['1']['cf'], conv_hparams['1']['hf'], conv_hparams['1']['wf'], conv_hparams['1']['pad'], conv_hparams['1']['stride'])
#        self.layers['BN_3'] = batch_normalization(n['0_3'], minibatch_size) # simple cnn
#        self.layers['BN_3'] = batch_normalization_book(np.ones((1,n['0_3']),'float32'),np.zeros((1,n['0_3']),'float32')) # simple cnn
        self.layers['actfunc_1'] = relu() # simple cnn
        self.layers['pool_1'] = pooling(pool_hparams['1']['hf'], pool_hparams['1']['wf'], pool_hparams['1']['stride'])
        self.layers['affine_2'] = affine(n['1'], n['2'])
#        self.layers['BN_4'] = batch_normalization(n['4'], minibatch_size)
#        self.layers['BN_4'] = batch_normalization_book(np.ones((1,n['4']),'float32'),np.zeros((1,n['4']),'float32'))
        self.layers['actfunc_2'] = relu()
#        self.layers['dropout_4'] = dropout(dropout_ratio)
        self.layers['affine_3'] = affine(n['2'], n['3'])
#        self.layers['dropout_5'] = dropout(dropout_ratio)
        self.layers['lastlayer'] = softmax_with_loss(minibatch_size)
        # パラメータのあるlayer
        self.layers_with_params = []
        words_list = ['affine', 'BN', 'conv']
        for key, layer in self.layers.items():
            for word in words_list:
                if word in key: self.layers_with_params.append(layer)
        for layer in self.layers_with_params:
            layer.W_optimizer = adam(opt_hparams['lr'], opt_hparams['weight_V'], opt_hparams['weight_H'])
            layer.b_optimizer = adam(opt_hparams['lr'], opt_hparams['weight_V'], opt_hparams['weight_H'])
        # フラグのあるlayer
        self.layers_with_flgs = []
        words_list = ['BN', 'dropout']
        for key, layer in self.layers.items():
            for word in words_list:
                if word in key: self.layers_with_flgs.append(layer)
    
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
        layers_list = list(self.layers.values())
        del layers_list[-1]
        for layer in layers_list:
            X = layer.forward(X)
        max_ind = np.argmax(X, axis=1).reshape([-1,1])
        acc = np.sum( max_ind==label ) / m * 100
        
        return acc
    
    def update_parameters(self):
        # パラメータのあるlayer
        for layer in self.layers_with_params:
            layer.W = layer.W_optimizer.update(layer.W, layer.dW)
            layer.b = layer.b_optimizer.update(layer.b, layer.db)
    
    def predict(self, X):
#        original_shape = X.shape
#        # 画像描画
#        X = X.reshape([28,28])
#        plt.imshow(X)
#        plt.colorbar()
#        plt.show()
#        # 推論
#        X = X.reshape(original_shape)
        layers_list = list(self.layers.values())
        del layers_list[-1]
        for layer in layers_list:
            X = layer.forward(X)
        max_ind = np.argmax(X, axis=1).reshape([-1,1])
#        print('predicted number: {}'.format(max_ind))
        
        return max_ind