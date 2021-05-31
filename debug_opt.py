# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from optimizers import *

W = np.random.rand(2, 3) # 0-1の一様分布
lr = 0.1
weight_V = 0.9
weight_H = 0.999
iters_num = 100

for iter in range(0, iters_num):
    if iter == 0:
        # opt
        opt = adam(lr, weight_V, weight_H)
        # ログ
        log_iter = np.zeros((iters_num, 1))
        log_loss = np.zeros((iters_num, 1))
    
    # grad
    loss = np.sum(W**2) # 仮の損失関数
    dW = 2*W
    
    # update
    W = opt.update(W, dW)
    
    # ログ
    log_iter[iter] = iter
    log_loss[iter] = loss

# グラフ
plt.plot(log_iter, log_loss)
plt.xlabel('iter')
plt.ylabel('loss')
plt.show()