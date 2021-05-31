# coding: utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
from network_cnn import network

# データ読込
print('data loading ...')
with open('mnist.pkl', 'rb') as f:
    dataset = pickle.load(f)

# 正規化
Data_train = dataset['train_img']
Data_test = dataset['test_img']
label_train = dataset['train_label'].reshape([60000,1])
label_test = dataset['test_label'].reshape([10000,1])
del dataset
Data_train = Data_train.astype('float32')
Data_test = Data_test.astype('float32')
Data_train /= 255.0 # .0は必要？
Data_test /= 255.0

# 分割
m = Data_train.shape[0]
m_train = int(m*4/5)
m_val = int(m/5)
m_test = Data_test.shape[0]
Data_val = Data_train[0:m_val, :]
Data_train = Data_train[m_val:m, :]
label_val = label_train[0:m_val, :]
label_train = label_train[m_val:m, :]

# 4次元に変換
Data_train = Data_train.reshape([m_train, 1, 28, 28])
Data_val = Data_val.reshape([m_val, 1, 28, 28])
Data_test = Data_test.reshape([m_test, 1, 28, 28])
m_train, c, h, w = Data_train.shape


# ハイパーパラメータ
# dropout
dropout_ratio = 0.5
# optimizer
lr = 0.001 # 学習率 learning rate
weight_V = 0.9
weight_H = 0.999
opt_hparams = {'lr':lr, 'weight_V':weight_V, 'weight_H':weight_H}
# iter
minibatch_size = 100 # 128
#minibatch_size = 4000 # batch
iter_per_epoch = m_train // minibatch_size
#iter_per_epoch = 1 # batch
epochs_num = 2
iters_num = iter_per_epoch * epochs_num
# 評価数
evaluation_num = 1000

# convolution + pooling
# pool:サイズを半分
#conv_pool_params = 0 # nn
conv_1_hparams = {'mf':30, 'cf':c, 'hf':5, 'wf':5, 'pad':0, 'stride':1} # simple cnn
pool_1_hparams = {'hf':2, 'wf':2, 'stride':2} # simple cnn
conv_hparams = {'1':conv_1_hparams}
pool_hparams = {'1':pool_1_hparams}
# h,w 出力サイズ
conv_1_ho = int(1 + (h + 2*conv_1_hparams['pad'] - conv_1_hparams['hf']) / conv_1_hparams['stride'])
conv_1_wo = int(1 + (w + 2*conv_1_hparams['pad'] - conv_1_hparams['wf']) / conv_1_hparams['stride'])
pool_1_ho = int(1 + (conv_1_ho - pool_1_hparams['hf']) / pool_1_hparams['stride'])
pool_1_wo = int(1 + (conv_1_wo - pool_1_hparams['wf']) / pool_1_hparams['stride'])

# layerサイズ
#n_3 = 28*28 # nn
n_0 = c * h * w
n0_1 = conv_1_hparams['mf'] * conv_1_ho * conv_1_wo # simple cnn
n_1 = conv_1_hparams['mf'] * pool_1_ho * pool_1_wo # simple cnn
n_2 = 50
n_3 = 10
#n = {'3':n_3, '4':n_4, '5':n_5} # nn
n = {'0':n_0, 
     '0_1':n0_1, '1':n_1, 
     '2':n_2, 
     '3':n_3} # simple cnn

# データサイズ
print('data size')
print(m_train, c, h, w)
print(m_train, conv_1_hparams['mf'], pool_1_ho, pool_1_wo)
print(m_train, n_2)
print(m_train, n_3)


# 学習
print('learning ...')
for iter in range(0, iters_num):
	   
    # 初期化
    if iter == 0:
        # ネットワーク
        nw = network(n, minibatch_size, 
                     conv_hparams, pool_hparams, 
                     dropout_ratio, 
                     opt_hparams)
        # 変数
        epoch = 0
        # ログ
        log_iter = np.zeros(iters_num, 'int')
        log_loss = np.zeros(iters_num)
        log_epoch = np.zeros(epochs_num, 'int')
        log_acc_train = np.zeros(epochs_num)
        log_acc_val = np.zeros(epochs_num)
    
    # ミニバッチ
    shuffled_ind = np.random.permutation(m_train)
    shuffled_ind = shuffled_ind[0:minibatch_size]
    Data_mini = Data_train[shuffled_ind[:], :]
    label_mini = label_train[shuffled_ind[:], :]
    # one-hot表現
    nw.layers['lastlayer'].Label = label_mini==np.arange(10).reshape([1,-1])
#    nw.layers['lastlayer'].Label = label_train==np.arange(10).reshape([1,-1]) # batch
    
    # grads
    for layer in nw.layers_with_flgs:
        layer.trained_flg = 0
    nw.get_gradients(Data_mini)
#    nw.get_gradients(Data_train) # batch
    for layer in nw.layers_with_flgs:
        layer.trained_flg = 1
    
    print('iter: {}, loss: {:.5f}'
          .format(iter, nw.layers['lastlayer'].loss))
    
    # パラメータの更新
    nw.update_parameters()
    
    # 推論
    if (iter+1) % iter_per_epoch == 0:
        # acc_val
        t = evaluation_num
        acc_train = nw.get_accuracy(Data_train[:t], label_train[:t])
        acc_val = nw.get_accuracy(Data_val[:t], label_val[:t])
#        acc_train = nw.get_accuracy(Data_train, label_train) # batch
#        acc_test = nw.get_accuracy(Data_test, label_test) # batch
        print('epoch: {}, acc_train: {:.5f}, acc_val: {:.5f}'
              .format(epoch, acc_train, acc_val))
        # ログ
        log_epoch[epoch] = epoch
        log_acc_train[epoch] = acc_train
        log_acc_val[epoch] = acc_val
        epoch += 1
    
    # ログ
    log_iter[iter] = iter
    log_loss[iter] = nw.layers['lastlayer'].loss


# acc_test
acc_test = nw.get_accuracy(Data_test, label_test)
print('acc_test: {:.5f}'.format(acc_test))


# ハイパーパラメータの保存
hparams = {}
hparams['n'] = n
hparams['minibatch_size'] = minibatch_size
hparams['conv_hparams'] = conv_hparams
hparams['pool_hparams'] = pool_hparams
hparams['dropout_ratio'] = dropout_ratio
hparams['opt_hparams'] = opt_hparams
with open('hparams.pkl', 'wb') as f:
    pickle.dump(hparams, f)
# パラメータの保存
params = {}
for i in range(1, 2):
    params['conv_'+str(i)+'_W'] = nw.layers['conv_'+str(i)].W
    params['conv_'+str(i)+'_b'] = nw.layers['conv_'+str(i)].b
params['affine_2_W'] = nw.layers['affine_2'].W
params['affine_2_b'] = nw.layers['affine_2'].b
params['affine_3_W'] = nw.layers['affine_3'].W
params['affine_3_b'] = nw.layers['affine_3'].b
with open('params.pkl', 'wb') as f:
    pickle.dump(params, f)


# グラフ
## loss
#plt.plot(log_iter, log_loss)
#plt.xlabel('iter')
#plt.ylabel('loss')
#plt.show()
# acc
plt.plot(log_epoch, log_acc_train, label='train')
plt.plot(log_epoch, log_acc_val, label='val')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()

## 推論
#import cv2
#Img = (cv2.imread('0.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255).reshape([1,1,28,28])
#nw.predict(Img)
#nw.predict(Data_train[0].reshape([1,-1]))
#Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

## ヒストグラム
#plt.hist(nw.layers['affine1'].W.reshape([-1,1]))

# プロファイラ
#import time
#start_time = time.time()
## 処理
#end_time = time.time()

