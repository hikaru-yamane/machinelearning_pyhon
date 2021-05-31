# coding: utf-8
import cv2
import pickle
from network import network

# hparams
with open('hparams_988.pkl', 'rb') as f:
    hparams = pickle.load(f)
n = hparams['n']
minibatch_size = hparams['minibatch_size']
conv_hparams = hparams['conv_hparams']
pool_hparams = hparams['pool_hparams']
dropout_ratio = hparams['dropout_ratio']
opt_hparams = hparams['opt_hparams']

# ネットワーク
nw = network(n, minibatch_size, 
             conv_hparams, pool_hparams, 
             dropout_ratio, 
             opt_hparams)

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

# Webカメラから入力を開始
cap = cv2.VideoCapture(0)
fps = 30
size = (640, 480) # 縦方向に反転させてから保存？
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('output.avi',fourcc,fps,size)

while True:
    # カメラの画像を読み込む サイズ(480,640,3) 座標(y,x,3)
    _, frame = cap.read()
    # 推論に使用する画像を切り取る
    box_one_side = 100
    left_top_x = int(frame.shape[1]/2 - box_one_side/2)
    left_top_y = int(frame.shape[0]/2 - box_one_side/2)
    right_bottom_x = int(frame.shape[1]/2 + box_one_side/2)
    right_bottom_y = int(frame.shape[0]/2 + box_one_side/2)
    # 注意: +1は不要
    Img = frame[left_top_y:right_bottom_y, left_top_x:right_bottom_x, :]
    # グレースケール
    Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    # 白黒反転
    Img = cv2.bitwise_not(Img)
    # 縮小
    Img = cv2.resize(Img, (28,28))
    # 正規化
    Img = Img.astype('float32') # 先に正規化するとエラー
    Img /= 255.0
    # 推論
    Img = Img.reshape([1,1,28,28])
    predicted_num = nw.predict(Img)
    # 枠
    left_top = (left_top_x, left_top_y)
    right_bottom = (right_bottom_x, right_bottom_y)
    color = (255,255,255)
    cv2.rectangle(frame,left_top,right_bottom,color)
    # 文字書き込み
    text = 'predicted number: ' + str(predicted_num)
    position = (40,400)
    font = cv2.FONT_ITALIC
    font_scale = 1
    color = (255,255,255)
    cv2.putText(frame,text,position,font,font_scale,color)
    # ウィンドウに画像を出力 --- (*4)
    cv2.imshow('OpenCV Web Camera', frame)
    # 書き込み
    video.write(frame)
    # ESCかEnterキーが押されたらループを抜ける
    k = cv2.waitKey(1) # 1msec確認
    if k == 27 or k == 13: break

cap.release() # カメラを解放
video.release() # 動画を解放
cv2.destroyAllWindows() # ウィンドウを破棄