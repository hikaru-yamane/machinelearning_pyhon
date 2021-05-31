# coding: utf-8
import numpy as np

class affine:
    def __init__(self, n_in, n_out):
#        # Xavierの初期値(sigmoid,tanh)
#        init_epsilon = np.sqrt(1/n_in)
        # Heの初期値(ReLU)
        # ニューロン１つが前層のいくつのニューロンとつながりがあるか
        connected_neurons_num = n_in
        init_epsilon = np.sqrt(2 / connected_neurons_num)
        self.W = np.random.randn(n_in, n_out) * init_epsilon
        self.W = self.W.astype('float32')
        self.b = np.zeros((1, n_out), 'float32')
        self.dW = 0
        self.db = 0
        self.W_optimizer = 0 # network.pyで値代入
        self.b_optimizer = 0 # network.pyで値代入
        # 計算過程で必要
        self.X_in_two = 0
        self.original_shape = 0
    
    def forward(self, X_in):
        # ２次元へ変換
        self.original_shape = X_in.shape
        self.X_in_two = X_in.reshape([X_in.shape[0],-1])
        
        X_out = np.dot(self.X_in_two, self.W) + self.b
        
        return X_out
    
    def backward(self, dX_out):
        dX_in_two = np.dot(dX_out, self.W.T)
        self.dW = np.dot(self.X_in_two.T, dX_out)
        self.db = np.sum(dX_out, axis=0, keepdims=True)
        
        # 元の次元へ変換
        dX_in = dX_in_two.reshape(self.original_shape)
        
        return dX_in


class sigmoid:
    def __init__(self):
        # 計算過程で必要
        self.X_out = 0
    
    def forward(self, X_in):
        self.X_out = 1 / (1 + np.exp(-X_in))
        
        return self.X_out
    
    def backward(self, dX_out):
        dX_in = dX_out * (self.X_out * (1-self.X_out))
        
        return dX_in


class relu:
    def __init__(self):
        # 計算過程で必要
        self.Mask = 0
    
    def forward(self, X_in):
        self.Mask = X_in >= 0
        X_out = X_in * self.Mask
        
        return X_out
    
    def backward(self, dX_out):
        dX_in = dX_out * self.Mask
        
        return dX_in


class sigmoid_with_loss:
    def __init__(self, m):
        self.loss = 0
        # 計算過程で必要
        self.m = m
        self.X_out = 0
        self.Label = 0 # main.pyで値代入
    
    def forward(self, X_in):
        self.X_out = 1 / (1 + np.exp(-X_in))
        # 交差エントロピー誤差 cross entropy error(sigmoid)
        self.loss = self.Label * np.log(self.X_out) \
                    + (1 - self.Label) * np.log(1 - self.X_out)
        self.loss = -np.sum(self.loss) / self.m
        
        return self.loss # debug:損失関数に注意
    
    def backward(self, dX_out):
        dX_in = (self.X_out - self.Label) / self.m
        
        return dX_in


class softmax_with_loss:
    def __init__(self, m):
        self.loss = 0
        # 計算過程で必要
        self.m = m
        self.epsilon = 1e-8
        self.X_out = 0
        self.Label = 0 # main.pyで値代入 # debug:条件に注意
    
    def forward(self, X_in):
        Temp = X_in - np.amax(X_in, axis=1, keepdims=True) # 指数のオーバーフロー対策
        self.X_out = np.exp(Temp) / np.sum(np.exp(Temp), axis=1, keepdims=True)
        self.X_out += self.epsilon # 対数のオーバーフロー対策
        # 交差エントロピー誤差 cross entropy error(relu)
        self.loss = -np.sum(self.Label * np.log(self.X_out)) / self.m
        
        return self.loss # debug:損失関数に注意
    
    def backward(self, dX_out):
        dX_in = (self.X_out - self.Label) / self.m
        
        return dX_in


class dropout:
    def __init__(self, dropout_ratio):
        # 計算過程で必要
        self.dropout_ratio = dropout_ratio
        self.Mask = 0
        self.original_shape = 0
        self.trained_flg = 0 # main.pyで値代入
    
    def forward(self, X_in):
        # ２次元へ変換
        self.original_shape = X_in.shape
        X_in_two = X_in.reshape([X_in.shape[0],-1])
        
        if self.trained_flg == 0:
            # debug:Mask保持に注意
            m, n = X_in_two.shape
            self.Mask = np.random.rand(m, n) > self.dropout_ratio
            X_out_two = X_in_two * self.Mask
        else:
            X_out_two = X_in_two * (1 - self.dropout_ratio)
        
        # 元の次元へ変換
        X_out = X_out_two.reshape(self.original_shape)
        
        return X_out
    
    def backward(self, dX_out):
        # ２次元へ変換
        dX_out_two = dX_out.reshape([dX_out.shape[0],-1])
        
        # trained_flg==1のときBPなし
        dX_in_two = dX_out_two * self.Mask
        
        # 元の次元へ変換
        dX_in = dX_in_two.reshape(self.original_shape)
        
        return dX_in


class batch_normalization_original:
    def __init__(self, n):
        self.W = np.ones((1, n)) # scale
        self.b = np.zeros((1, n)) # shift
        self.dW = 0
        self.db = 0
        self.W_optimizer = 0 # network.pyで値代入
        self.b_optimizer = 0 # network.pyで値代入
        # 計算過程で必要
        self.m = 0
        self.epsilon = 1e-8 # オーバーフロー対策 1e-8
        self.f1 = 0
        self.f3 = 0
        self.f6 = 0
        self.f7 = 0
        self.f8 = 0
        self.f9 = 0
        self.f11 = 0
        self.trained_flg = 0 # main.pyで値代入
    
    def forward(self, X_in):
        self.m = X_in.shape[0]
        
        f0 = X_in
        f1 = np.sum(f0, axis=0, keepdims=True) / self.m
        if self.trained_flg == 1: f1 = self.f1
        f2 = np.ones_like(X_in) * f1
        f3 = f0 - f2
        f4 = f3**2
        f5 = np.sum(f4, axis=0, keepdims=True) / self.m
        f6 = np.sqrt(f5 + self.epsilon)
        if self.trained_flg == 1: f6 = self.f6
        f7 = 1 / f6
        f8 = np.ones_like(X_in) * f7
        f9 = f3 * f8
        f10 = self.W
        f11 = np.ones_like(X_in) * f10
        f12 = f11 * f9
        f13 = self.b
        f14 = np.ones_like(X_in) * f13
        X_out = f12 + f14
        
        self.f1 = f1
        self.f3 = f3
        self.f6 = f6
        self.f7 = f7
        self.f8 = f8
        self.f9 = f9
        self.f11 = f11
        
        return X_out
    
    def backward(self, dX_out):
        d15 = dX_out
        d14 = np.sum(d15, axis=0, keepdims=True)
        d12a = d15 * self.f11
        d12b = d15 * self.f9
        d11 = np.sum(d12b, axis=0, keepdims=True)
        d9a = d12a * self.f8
        d9b = d12a * self.f3
        d8 = np.sum(d9b, axis=0, keepdims=True)
        d7 = -d8 * (self.f7**2) # 念のため(-f7**2)は避けとく
        d6 = d7 * (1/(2*self.f6)) # d7/(2*f6) でもいいが可読性重視
        d5 = np.ones_like(dX_out) * d6 / self.m
        d4 = d5 * (2*self.f3)
        d3a = d9a + d4
        d3b = -d3a
        d2 = np.sum(d3b, axis=0, keepdims=True)
        d1 = np.ones_like(dX_out) * d2 / self.m
        dX_in = d3a + d1
        
        self.dW = d11
        self.db = d14
        
        return dX_in


class batch_normalization:
    def __init__(self, n, m):
        self.W = np.ones((1, n), 'float32') # scale
#        self.W = np.ones((1, n))
        self.b = np.zeros((1, n), 'float32') # shift
#        self.b = np.zeros((1, n))
        self.dW = 0
        self.db = 0
        self.W_optimizer = 0 # network.pyで値代入
        self.b_optimizer = 0 # network.pyで値代入
        # 計算過程で必要
        self.m = m
        self.epsilon = 1e-8 # オーバーフロー対策
        self.X_mu = 0
        self.mu = 0 # 平均 mean (統計学ではミューmu)
        self.std = 0 # 標準偏差 standard deviation
        self.original_shape = 0
        self.trained_flg = 0 # main.pyで値代入
        self.mu_predict = 0
        self.std_predict = 0
    
    def forward(self, X_in):
        # ２次元へ変換
        self.original_shape = X_in.shape
        X_in_two = X_in.reshape([X_in.shape[0],-1])
        
        # 平均と標準偏差
        if self.trained_flg == 0:
            self.mu = np.mean(X_in_two, axis=0, keepdims=True)
            self.X_mu = X_in_two - self.mu
            var = np.var(X_in_two, axis=0, keepdims=True)
            self.std = np.sqrt(var + self.epsilon)
            
            self.mu_predict = 0.9 * self.mu_predict + 0.1 * self.mu
            self.std_predict = 0.9 * self.std_predict + 0.1 * self.std
        else:
#            self.X_mu = X_in_two - self.mu
            self.X_mu = X_in_two - self.mu_predict
            self.std = self.std_predict
        
        X_out_two = self.W * self.X_mu / self.std + self.b
        
        # 元の次元へ変換
        X_out = X_out_two.reshape(self.original_shape)
        
        return X_out
    
    def backward(self, dX_out):
        # ２次元へ変換
        dX_out_two = dX_out.reshape([dX_out.shape[0],-1])
        
        d9a = dX_out_two * self.W / self.std
        Temp = dX_out_two * self.W * self.X_mu
        Temp = np.sum(Temp, axis=0, keepdims=True)
        Temp = -Temp * self.X_mu / (self.m * self.std**3)
        Temp = d9a + Temp
        dX_in_two = Temp - np.sum(Temp, axis=0, keepdims=True) / self.m
        
        self.dW = np.sum(dX_out_two * self.X_mu / self.std, axis=0, keepdims=True)
        self.db = np.sum(dX_out_two, axis=0, keepdims=True)
        
        # 元の次元へ変換
        dX_in = dX_in_two.reshape(self.original_shape)
        
        return dX_in


class batch_normalization_book:
    def __init__(self, W, b, momentum=0.9, running_mean=None, running_var=None):
        self.W = W
        self.b = b
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  
        self.W_optimizer = 0 # network.pyで値代入
        self.b_optimizer = 0 # network.pyで値代入

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dW = None
        self.db = None
        self.trained_flg = 0 # main.pyで値代入

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        
#        if train_flg:
        if self.trained_flg == 0:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-8)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-8)))
            
        out = self.W * xn + self.b
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        db = dout.sum(axis=0)
        dW = np.sum(self.xn * dout, axis=0)
        dxn = self.W * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dW = dW
        self.db = db
        
        return dx


class convolution:
    def __init__(self, mf, c, hf, wf, pad, stride):
        # Heの初期値(ReLU)
        # ニューロン１つが前層のいくつのニューロンとつながりがあるか
        connected_neurons_num = c * hf * wf
        init_epsilon = np.sqrt(2 / connected_neurons_num)
        self.W = np.random.randn(mf, c, hf, wf) * init_epsilon
        self.W = self.W.astype('float32')
        self.b = np.zeros((1, mf), 'float32')
        self.dW = 0
        self.db = 0
        self.W_optimizer = 0 # network.pyで値代入
        self.b_optimizer = 0 # network.pyで値代入
        # 計算過程で必要
        self.pad = pad
        self.stride = stride
        self.X_in_two = 0
        self.original_shape = 0
    
    def forward(self, X_in):
        # 初期化
        m, c, h, w = X_in.shape
        mf, c, hf, wf = self.W.shape # filter
        ho = int(1 + (h + 2*self.pad - hf) / self.stride) # output
        wo = int(1 + (w + 2*self.pad - wf) / self.stride)
        
        # four2two
        # (m,c,h,w) →(m,c,ho,wo,hf,wf) →(m,ho,wo,c,hf,wf) →(m*ho*wo,c*hf*wf)
        X_in_six = four2six_2(X_in, hf, wf, self.pad, self.stride) # (m,c,ho,wo,hf,wf)
        X_in_six = X_in_six.transpose(0,2,3,1,4,5) # (m,ho,wo,c,hf,wf)
        self.X_in_two = X_in_six.reshape([m*ho*wo, c*hf*wf])
        W_two = self.W.reshape([mf, c*hf*wf]).T
        
        # affine.forward
        X_out_two = np.dot(self.X_in_two, W_two) + self.b # (m*ho*wo,mf)
        
        # two2four
        X_out = X_out_two.reshape([m, ho, wo, mf])
        X_out = X_out.transpose(0,3,1,2) # (m,mf,ho,wo)
        
        # インスタンス変数
        self.original_shape = (m, c, h, w)
        
        return X_out
    
    def backward(self, dX_out):
        # 初期化
        m, c, h, w = self.original_shape
        mf, c, hf, wf = self.W.shape
        ho = int(1 + (h + 2*self.pad - hf) / self.stride)
        wo = int(1 + (w + 2*self.pad - wf) / self.stride)
        W_two = self.W.reshape([mf, c*hf*wf]).T
        
        # four2two
        dX_out = dX_out.transpose(0,2,3,1) # (m,ho,wo,mf)
        dX_out_two = dX_out.reshape([m*ho*wo, mf])
        
        # affine.backward
        dX_in_two = np.dot(dX_out_two, W_two.T) # (m*ho*wo,c*hf*wf)
        dW_two = np.dot(self.X_in_two.T, dX_out_two) # (c*hf*wf,mf)
        self.db = np.sum(dX_out_two, axis=0, keepdims=True) # (1,mf)
        
        # two2four
        dX_in_six = dX_in_two.reshape([m, ho, wo, c, hf, wf])
        dX_in_six = dX_in_six.transpose(0,3,1,2,4,5) # (m,c,ho,wo,hf,wf)
        dX_in = six2four_2(dX_in_six, self.original_shape, hf, wf, self.pad, self.stride) # (m,c,h,w)
        self.dW = dW_two.T.reshape([mf, c, hf, wf])
        
        return dX_in


class pooling:
    def __init__(self, hf, wf, stride):
        # 計算過程で必要
        self.hf = hf
        self.wf = wf
        self.pad = 0
        self.stride = stride
        self.Mask = 0
        self.original_shape = 0
    
    def forward(self, X_in):
        # 初期化
        m, c, h, w = X_in.shape
        hf = self.hf # filter
        wf = self.wf
        ho = int(1 + (h - hf) / self.stride) # output
        wo = int(1 + (w - wf) / self.stride)
        
        # four2two
        # (m,c,h,w) →(m,c,ho,wo,hf,wf) →(m*c*ho*wo, hf*wf)
        X_in_six = four2six_2(X_in, hf, wf, self.pad, self.stride) # (m,c,ho,wo,hf,wf)
        X_in_two = X_in_six.reshape([m*c*ho*wo, hf*wf])
        
        # max_pooling.forward
        X_out_two = np.amax(X_in_two, axis=1, keepdims=True) # (m*c*ho*wo,1)
        self.Mask = X_in_two==X_out_two # (m*c*ho*wo,hf*wf)
        
        # two2four
        X_out = X_out_two.reshape([m, c, ho, wo])
        
        # インスタンス変数
        self.original_shape = (m, c, h, w)
        
        return X_out
    
    def backward(self, dX_out):
        # 初期化
        m, c, h, w = self.original_shape
        hf = self.hf
        wf = self.wf
        ho = int(1 + (h - hf) / self.stride)
        wo = int(1 + (w - wf) / self.stride)
        
        # four2two
        dX_out_two = dX_out.reshape([m*c*ho*wo, 1])
        
        # max_pooling.backward
        dX_in_two = self.Mask * dX_out_two # (m*c*ho*wo,hf*wf)
        
        # two2four
        dX_in_six = dX_in_two.reshape([m, c, ho, wo, hf, wf])
        dX_in = six2four_2(dX_in_six, self.original_shape, hf, wf, self.pad, self.stride) # (m,c,h,w)
        
        return dX_in


def four2six_1(X_four, hf, wf, pad, stride):
    # 初期化
    m, c, h, w = X_four.shape
    ho = int(1 + (h + 2*pad - hf) / stride)
    wo = int(1 + (w + 2*pad - wf) / stride)
    
    # pad
    X_four_with_pad = np.pad(X_four, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    
    # four2six
    X_six = np.zeros((m, c, ho, wo, hf, wf), 'float32')
    for i_filter in range(0, ho):
        for j_filter in range(0, wo):
            for i_cell in range(0, hf):
                for j_cell in range(0, wf):
                    # (m,c)
                    X_six[:,:,i_filter,j_filter,i_cell,j_cell] = X_four_with_pad[:, :, i_filter * stride + i_cell, j_filter * stride + j_cell]
    
    return X_six


def six2four_1(dX_six, original_shape, hf, wf, pad, stride):
    # 初期化
    m, c, h, w = original_shape
    ho = int(1 + (h + 2*pad - hf) / stride)
    wo = int(1 + (w + 2*pad - wf) / stride)
    
    # six2four
    dX_four_with_pad = np.zeros((m, c, h + 2*pad, w + 2*pad), 'float32')
    for i_filter in range(0, ho):
        for j_filter in range(0, wo):
            for i_cell in range(0, hf):
                for j_cell in range(0, wf):
                    # (m,c)
                    dX_four_with_pad[:, :, i_filter * stride + i_cell, j_filter * stride + j_cell] += dX_six[:,:,i_filter,j_filter,i_cell,j_cell]
    
    # pad
    dX_four = dX_four_with_pad[:, :, pad:pad + h, pad:pad + w]
    
    return dX_four


def four2six_2(X_four, hf, wf, pad, stride):
    # 初期化
    m, c, h, w = X_four.shape
    ho = int(1 + (h + 2*pad - hf) / stride)
    wo = int(1 + (w + 2*pad - wf) / stride)
    
    # pad
    X_four_with_pad = np.pad(X_four, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    
    # four2six
    X_six = np.zeros((m, c, ho, wo, hf, wf), 'float32')
    for i_cell in range(0, hf):
        for j_cell in range(0, wf):
            # (m,c,ho,wo)
            i_max = i_cell + (ho-1) * stride + 1 # :(終わりの添字)+1
            j_max = j_cell + (wo-1) * stride + 1
            X_six[:,:,:,:,i_cell,j_cell] = X_four_with_pad[:, :, i_cell:i_max:stride, j_cell:j_max:stride]
    
    return X_six


def six2four_2(dX_six, original_shape, hf, wf, pad, stride):
    # 初期化
    m, c, h, w = original_shape
    ho = int(1 + (h + 2*pad - hf) / stride)
    wo = int(1 + (w + 2*pad - wf) / stride)
    
    # six2four
    dX_four_with_pad = np.zeros((m, c, h + 2*pad, w + 2*pad), 'float32')
    for i_cell in range(0, hf):
        for j_cell in range(0, wf):
            # (m,c,ho,wo)
            i_max = i_cell + (ho-1) * stride + 1
            j_max = j_cell + (wo-1) * stride + 1
            dX_four_with_pad[:, :, i_cell:i_max:stride, j_cell:j_max:stride] += dX_six[:,:,:,:,i_cell,j_cell]
    
    # pad
    dX_four = dX_four_with_pad[:, :, pad:pad + h, pad:pad + w]
    
    return dX_four


def four2six_3(X_four, hf, wf, pad, stride):
    # 初期化
    m, c, h, w = X_four.shape
    ho = int(1 + (h + 2*pad - hf) / stride)
    wo = int(1 + (w + 2*pad - wf) / stride)
    
    # pad
    X_four_with_pad = np.pad(X_four, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    
    # four2six
    X_six = np.zeros((m, c, ho, wo, hf, wf), 'float32')
    for i_filter in range(0, ho):
        for j_filter in range(0, wo):
            # (m,c,hf,wf)
            i_min = i_filter * stride
            j_min = j_filter * stride
            X_six[:,:,i_filter,j_filter,:,:] = X_four_with_pad[:, :, i_min:i_min + hf, j_min:j_min + wf]
    
    return X_six


def six2four_3(dX_six, original_shape, hf, wf, pad, stride):
    # 初期化
    m, c, h, w = original_shape
    ho = int(1 + (h + 2*pad - hf) / stride)
    wo = int(1 + (w + 2*pad - wf) / stride)
    
    # six2four
    dX_four_with_pad = np.zeros((m, c, h + 2*pad, w + 2*pad), 'float32')
    for i_filter in range(0, ho):
        for j_filter in range(0, wo):
            # (m,c,hf,wf)
            i_min = i_filter * stride
            j_min = j_filter * stride
            dX_four_with_pad[:, :, i_min:i_min + hf, j_min:j_min + wf] += dX_six[:,:,i_filter,j_filter,:,:]
    
    # pad
    dX_four = dX_four_with_pad[:, :, pad:pad + h, pad:pad + w]
    
    return dX_four