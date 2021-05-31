# coding: utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
from network_job import network


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
Data_train /= 255.0
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
iter_per_epoch = m_train // minibatch_size
epochs_num = 3
iters_num = iter_per_epoch * epochs_num
# 評価数
evaluation_num = 1000

# convolution + pooling
# #conv:サイズを固定, pool:サイズを半分
conv0_1_hparams = {'mf':16, 'cf':c, 'hf':3, 'wf':3, 'pad':1, 'stride':1}
conv1_1_hparams = {'mf':16, 'cf':conv0_1_hparams['mf'], 'hf':3, 'wf':3, 'pad':1, 'stride':1}
pool_1_hparams = {'hf':2, 'wf':2, 'stride':2}
conv0_2_hparams = {'mf':32, 'cf':conv1_1_hparams['mf'], 'hf':3, 'wf':3, 'pad':1, 'stride':1}
conv1_2_hparams = {'mf':32, 'cf':conv0_2_hparams['mf'], 'hf':3, 'wf':3, 'pad':1, 'stride':1}
pool_2_hparams = {'hf':2, 'wf':2, 'stride':2}
conv0_3_hparams = {'mf':64, 'cf':conv1_2_hparams['mf'], 'hf':3, 'wf':3, 'pad':1, 'stride':1}
conv1_3_hparams = {'mf':64, 'cf':conv0_3_hparams['mf'], 'hf':3, 'wf':3, 'pad':1, 'stride':1}
pool_3_hparams = {'hf':2, 'wf':2, 'stride':2}
conv_hparams = {'0_1':conv0_1_hparams, '1_1':conv1_1_hparams, 
                '0_2':conv0_2_hparams, '1_2':conv1_2_hparams, 
                '0_3':conv0_3_hparams, '1_3':conv1_3_hparams}
pool_hparams = {'1':pool_1_hparams, 
                '2':pool_2_hparams, 
                '3':pool_3_hparams}

# h,w 出力サイズ
conv0_1_ho = int(1 + (h + 2*conv0_1_hparams['pad'] - conv0_1_hparams['hf']) / conv0_1_hparams['stride'])
conv0_1_wo = int(1 + (w + 2*conv0_1_hparams['pad'] - conv0_1_hparams['wf']) / conv0_1_hparams['stride'])
conv1_1_ho = int(1 + (conv0_1_ho + 2*conv1_1_hparams['pad'] - conv1_1_hparams['hf']) / conv1_1_hparams['stride'])
conv1_1_wo = int(1 + (conv0_1_wo + 2*conv1_1_hparams['pad'] - conv1_1_hparams['wf']) / conv1_1_hparams['stride'])
pool_1_ho = int(1 + (conv1_1_ho - pool_1_hparams['hf']) / pool_1_hparams['stride'])
pool_1_wo = int(1 + (conv1_1_wo - pool_1_hparams['wf']) / pool_1_hparams['stride'])
conv0_2_ho = int(1 + (pool_1_ho + 2*conv0_2_hparams['pad'] - conv0_2_hparams['hf']) / conv0_2_hparams['stride'])
conv0_2_wo = int(1 + (pool_1_wo + 2*conv0_2_hparams['pad'] - conv0_2_hparams['wf']) / conv0_2_hparams['stride'])
conv1_2_ho = int(1 + (conv0_2_ho + 2*conv1_2_hparams['pad'] - conv1_2_hparams['hf']) / conv1_2_hparams['stride'])
conv1_2_wo = int(1 + (conv0_2_wo + 2*conv1_2_hparams['pad'] - conv1_2_hparams['wf']) / conv1_2_hparams['stride'])
pool_2_ho = int(1 + (conv1_2_ho - pool_2_hparams['hf']) / pool_2_hparams['stride'])
pool_2_wo = int(1 + (conv1_2_wo - pool_2_hparams['wf']) / pool_2_hparams['stride'])
conv0_3_ho = int(1 + (pool_2_ho + 2*conv0_3_hparams['pad'] - conv0_3_hparams['hf']) / conv0_3_hparams['stride'])
conv0_3_wo = int(1 + (pool_2_wo + 2*conv0_3_hparams['pad'] - conv0_3_hparams['wf']) / conv0_3_hparams['stride'])
conv1_3_ho = int(1 + (conv0_3_ho + 2*conv1_3_hparams['pad'] - conv1_3_hparams['hf']) / conv1_3_hparams['stride'])
conv1_3_wo = int(1 + (conv0_3_wo + 2*conv1_3_hparams['pad'] - conv1_3_hparams['wf']) / conv1_3_hparams['stride'])
pool_3_ho = int(1 + (conv1_3_ho - pool_3_hparams['hf']) / pool_3_hparams['stride'])
pool_3_wo = int(1 + (conv1_3_wo - pool_3_hparams['wf']) / pool_3_hparams['stride'])

# layerサイズ
n_0 = c * h * w
n0_1 = conv0_1_hparams['mf'] * conv0_1_ho * conv0_1_wo
n1_1 = conv1_1_hparams['mf'] * conv1_1_ho * conv1_1_wo
n_1 = conv1_1_hparams['mf'] * pool_1_ho * pool_1_wo
n0_2 = conv0_2_hparams['mf'] * conv0_2_ho * conv0_2_wo
n1_2 = conv1_2_hparams['mf'] * conv1_2_ho * conv1_2_wo
n_2 = conv1_2_hparams['mf'] * pool_2_ho * pool_2_wo
n0_3 = conv0_3_hparams['mf'] * conv0_3_ho * conv0_3_wo
n1_3 = conv1_3_hparams['mf'] * conv1_3_ho * conv1_3_wo
n_3 = conv1_3_hparams['mf'] * pool_3_ho * pool_3_wo
n_4 = 50
n_5 = 10
n = {'0':n_0, 
     '0_1':n0_1, '1_1':n1_1, '1':n_1, 
     '0_2':n0_2, '1_2':n1_2, '2':n_2, 
     '0_3':n0_3, '1_3':n1_3, '3':n_3, 
     '4':n_4, 
     '5':n_5}

# データサイズ
print('data size')
print(m_train, c, h, w)
print(m_train, conv0_1_hparams['mf'], conv0_1_ho, conv0_1_wo)
print(m_train, conv1_1_hparams['mf'], conv1_1_ho, conv1_1_wo)
print(m_train, conv1_1_hparams['mf'], pool_1_ho, pool_1_wo)
print(m_train, conv0_2_hparams['mf'], conv0_2_ho, conv0_2_wo)
print(m_train, conv1_2_hparams['mf'], conv1_2_ho, conv1_2_wo)
print(m_train, conv1_2_hparams['mf'], pool_2_ho, pool_2_wo)
print(m_train, conv0_3_hparams['mf'], conv0_3_ho, conv0_3_wo)
print(m_train, conv1_3_hparams['mf'], conv1_3_ho, conv1_3_wo)
print(m_train, conv1_3_hparams['mf'], pool_3_ho, pool_3_wo)
print(m_train, n_4)
print(m_train, n_5)


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
    
    # grads
    for layer in nw.layers_with_flgs:
        layer.trained_flg = 0
    nw.get_gradients(Data_mini)
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
for i in range(1, 4):
    params['conv0_'+str(i)+'_W'] = nw.layers['conv0_'+str(i)].W
    params['conv0_'+str(i)+'_b'] = nw.layers['conv0_'+str(i)].b
    params['conv1_'+str(i)+'_W'] = nw.layers['conv1_'+str(i)].W
    params['conv1_'+str(i)+'_b'] = nw.layers['conv1_'+str(i)].b
params['affine_4_W'] = nw.layers['affine_4'].W
params['affine_4_b'] = nw.layers['affine_4'].b
params['affine_5_W'] = nw.layers['affine_5'].W
params['affine_5_b'] = nw.layers['affine_5'].b
with open('params.pkl', 'wb') as f:
    pickle.dump(params, f)


# グラフ
# loss
plt.plot(log_iter, log_loss)
plt.xlabel('iter')
plt.ylabel('loss')
plt.show()
# acc
plt.plot(log_epoch, log_acc_train, label='train')
plt.plot(log_epoch, log_acc_val, label='val')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()
