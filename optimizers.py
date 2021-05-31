# coding: utf-8
import numpy as np

# HやV保持のため，関数ではなくクラス使用

class gradient_descent:
    def __init__(self, lr):
        self.lr = lr # 0.01
    
    def update(self, X_in, dX_in):
        X_out = X_in - self.lr * dX_in
        
        return X_out


class momentum:
    def __init__(self, lr, weight_V):
        self.lr = lr # 0.01
        self.weight_V = weight_V # 0.9
        self.V = 0
    
    def update(self, X_in, dX_in):
        self.V = self.weight_V * self.V - self.lr * dX_in
        X_out = X_in + self.V
        
        return X_out


class adagrad:
    def __init__(self, lr):
        self.lr = lr # 0.001
        self.H = 0
        self.epsilon = 1e-8
    
    def update(self, X_in, dX_in):
        self.H += dX_in**2
        X_out = X_in - self.lr * dX_in / np.sqrt(self.H + self.epsilon)
        
        return X_out


class rmsprop:
    def __init__(self, lr, weight_H):
        self.lr = lr # 0.01
        self.weight_H = weight_H # 0.99
        self.H = 0
        self.epsilon = 1e-8
    
    def update(self, X_in, dX_in):
        self.H = self.weight_H * self.H + (1-self.weight_H) * (dX_in**2)
        X_out = X_in - self.lr * dX_in / np.sqrt(self.H + self.epsilon)
        
        return X_out


class adam:
    def __init__(self, lr, weight_V, weight_H):
        self.lr = lr # 0.001
        self.weight_V = weight_V # 0.9
        self.weight_H = weight_H # 0.999
        self.V = 0
        self.H = 0
        self.cnt = 1
        self.epsilon = 1e-8
    
    def update(self, X_in, dX_in):
        self.V = self.weight_V * self.V + (1-self.weight_V) * dX_in
        self.H = self.weight_H * self.H + (1-self.weight_H) * (dX_in**2)
        
        V_hat = self.V / (1 - self.weight_V**self.cnt)
        H_hat = self.H / (1 - self.weight_H**self.cnt)
        
        X_out = X_in - self.lr * V_hat / np.sqrt(H_hat + self.epsilon)
        
        self.cnt += 1 # iterを持ってくるのは大変だから計算する
        
        return X_out