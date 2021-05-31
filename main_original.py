# coding: utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
from network_original import network


# データ読込
print('data loading ...')
with open('mnist.pkl', 'rb') as f:
     dataset = pickle.load(f)
# 結合・正規化・シャッフル
Data = np.concatenate([dataset['train_img'], 
																							dataset['test_img']], axis=0) # 縦に結合
label = np.concatenate([dataset['train_label'].reshape([60000,1]), 
																								dataset['test_label'].reshape([10000,1])], axis=0)
del dataset # メモリ解放
Data = Data.astype('float32') # 意味ある？
Data /= 255.0 # .0は必要？
m = Data.shape[0]
shuffled_ind = np.random.permutation(m) # １次元配列だけど計算に無関係だからおｋ
# データ分割
m_train = int(m*6/7) # intがないと.0が入り整数でなくなる
m_test = int(m/7)
m = 5000
m_train = 4000
m_test = 1000
Data_train = Data[shuffled_ind[0:m_train], :]
Data_test = Data[shuffled_ind[m_train:m], :]
label_train = label[shuffled_ind[0:m_train], :]
label_test = label[shuffled_ind[m_train:m], :]
del Data, label, shuffled_ind


# nn構成
n0 = 784 # 28*28
n1 = 25
n2 = 10


# ハイパーパラメータ
lr = 0.01 # 学習率 learning rate
# dropout
dropout_ratio = 0.5
# optimizer
weight_V = 0.9
weight_H = 0.999
# iter
minibatch_size = 128
iter_per_epoch = m_train // minibatch_size # 1エポックあたりの回数
epochs_num = 10
iters_num = iter_per_epoch * epochs_num


# 学習
print('learning ...')
for iter in range(0, iters_num):
	   
    # 初期化
    if iter == 0:
        # ネットワーク
        nw = network(n0, n1, n2, minibatch_size, 
                     lr, weight_V, weight_H, dropout_ratio)
        # 変数
        epoch = 0
        # ログ
        log_iter = np.zeros((iters_num, 1))
        log_loss = np.zeros((iters_num, 1))
        log_epoch = np.zeros((epochs_num, 1))
        log_acc_train = np.zeros((epochs_num, 1))
        log_acc_test = np.zeros((epochs_num, 1))
    
    # ミニバッチ
    shuffled_ind = np.random.permutation(m_train)
    shuffled_ind = shuffled_ind[0:minibatch_size]
    Data_mini = Data_train[shuffled_ind[:], :]
    label_mini = label_train[shuffled_ind[:], :]
    # one-hot表現に変換
    # bool型だけどintのサブクラスだからそのままでいいらしい
    nw.layers['lastlayer'].Label = label_mini==np.arange(10).reshape([1,-1])
    
    # FP & BP
    # grads
    # インスタンス変数を多用すると可読性が低下するが入出力が大変だからここはおｋ
#    nw.layers['BN1'].trained_flg = 0
#    nw.layers['dropout1'].trained_flg = 0
    nw.get_gradients(Data_mini)
#    nw.layers['BN1'].trained_flg = 1
#    nw.layers['dropout1'].trained_flg = 1
    
    print('iter: {}, loss: {:.5f}'.format(iter, nw.layers['lastlayer'].loss))
    
    # 推論
    if (iter+1) % iter_per_epoch == 0:
        # acc
        # このように極力関数を使うと分かりやすい(書く量は増える)
        acc_train = nw.get_accuracy(Data_train, label_train)
        acc_test = nw.get_accuracy(Data_test, label_test)
        print('epoch: {}, acc_train: {:.5f}, acc_test: {:.5f}'
              .format(epoch, acc_train, acc_test))
        # ログ
        log_epoch[epoch] = epoch
        log_acc_train[epoch] = acc_train
        log_acc_test[epoch] = acc_test
        epoch += 1
    
    # パラメータの更新 params
    nw.update_parameters()
    
    # ログ
    log_iter[iter] = iter
    log_loss[iter] = nw.layers['lastlayer'].loss
    

# パラメータ・ハイパーパラメータの保存

# グラフ
## loss 学習曲線
#plt.plot(log_iter, log_loss)
#plt.xlabel('iter')
#plt.ylabel('loss')
#plt.show()
# acc
plt.plot(log_epoch, log_acc_train, label='train')
plt.plot(log_epoch, log_acc_test, label='test')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.show()

## 推論
#import cv2
#Img = cv2.imread('0.png', cv2.IMREAD_GRAYSCALE)
#Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
#nw.predict(Img)
#nw.predict(Data_train[0])

## ヒストグラム
#plt.hist(nw.layers['affine1'].W.reshape([-1,1]))
# 実行時間計測
