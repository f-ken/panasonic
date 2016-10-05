#### 機械学習基礎講座第3回演習サンプルプログラム ex3_2.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2016/09/23

#### 多層パーセプトロンによる手書き文字認識
#### Python機械学習本：12.2.1, 12.2.2節 (pp. 333-345)

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP
import sys

# MNISTデータの読み込み関数
def load_mnist(path, kind='train'):

    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
   
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        
    return images, labels

# MNISTデータの読み込み
current_path = os.path.dirname(os.path.realpath(__file__))
X_train, y_train = load_mnist(current_path, kind='train')
X_test, y_test = load_mnist(current_path, kind='t10k')

# =====================================================================
# 課題3(a): 学習用データ(trn)はX_trainの最初の1000点，検証用データ(vld)はX_trainの
# 次の300点，テスト用データ(tst)はX_testの最初の300点をそれぞれX_trn, X_vld, X_tstに
# 格納する．クラスラベルは対応するy_*に格納する．

n_training_data = 1000
n_validation_data = 300
n_test_data = 300

X_trn = X_train[:n_training_data][:]
y_trn = y_train[:n_training_data][:]
X_vld = X_train[n_training_data+1:n_training_data+n_validation_data][:]
y_vld = y_train[n_training_data+1:n_training_data+n_validation_data][:]
X_tst = X_test[:n_test_data][:]
y_tst = y_test[:n_test_data][:]

# =====================================================================
# 多層パーセプトロン(MLP)のインスタンスの生成と学習

nn = NeuralNetMLP(n_output=10,                # 出力ユニット数
                  n_features=X_trn.shape[1],  # 入力ユニット数
                  n_hidden=30,                # 隠れユニット数
                  l2=0.1,                     # L2正則化のλパラメータ
                  l1=0.5,                     # L1正則化のλパラメータ
                  epochs=600,                 # 学習エポック数
                  eta=0.001,                  # 学習率の初期値
                  alpha = 0.001,              # モーメンタム学習の1つ前の勾配の係数
                  decrease_const=0.00001,     # 適応学習率の減少定数
                  minibatches=50,             # 各エポックでのミニバッチ数
                  shuffle=True,               # データのシャッフル
                  random_state=3)             # 乱数シードの状態
                  
nn.fit(X_trn, y_trn, print_progress=True)

plt.figure(0)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 1000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()

# =====================================================================
# 課題3(b): 学習用データ，検証用データおよびテストデータに対するaccuracyの算出を追加する

y_trn_pred = nn.predict(X_trn)
acc = np.sum(y_trn == y_trn_pred, axis=0) / X_trn.shape[0]
print('accuracy for trn data: %.2f%%' % (acc * 100))

y_vld_pred = nn.predict(X_vld)
acc = np.sum(y_vld == y_vld_pred, axis=0) / X_vld.shape[0]
print('accuracy for vld data: %.2f%%' % (acc * 100))

y_tst_pred = nn.predict(X_tst)
acc = np.sum(y_tst == y_tst_pred, axis=0) / X_tst.shape[0]
print('accuracy for tst data: %.2f%%' % (acc * 100))

# =====================================================================

plt.show()
