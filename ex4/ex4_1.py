#### 機械学習基礎講座第4回演習サンプルプログラム ex4_1.py
#### Programmed by Wu Hongle, 監修　福井健一
#### Last updated: 2016/10/04

#### SVMによるBreast Cancerデータの識別
#### Python機械学習本：3.5.1節，4.5.2節
#### Breast Cancerデータについては下記を参照
#### https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

import numpy as np
from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Breast Cancerデータのロード
df = load_breast_cancer()
X = df.data
y = df.target

# z標準化
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# =====================================================================
# 外側ループのための交差検証用データの用意
# 注）本プログラムでは，チューニングしたパラメータを出力するためにcross_val_score()は使用しない
kfold = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=2)

turned_params = [] #外側ループのfold毎にチューニングしたパラメータ格納用
acc_trn_list = []  #外側ループのfold毎の学習データに対するaccuracy格納用
acc_tst_list = []  #外側ループのfold毎のテストデータに対するaccuracy格納用
parameters = {'gamma':[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]} #グリッドサーチのパラメータリスト
gs = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=2) #内側ループのグリッドサーチを行う交差検証インスタンス

for k, (train_itr, test_itr) in enumerate(kfold):
	# 内側ループのグリッドサーチ  
    gs.fit(X[train_itr], y[train_itr])
    print('Fold #{:2d}; Best Parameter: {:1.3f}, Accuracy on validation data: {:.3f}'\
        .format(k+1,gs.best_params_['gamma'],gs.best_score_))

    #==========================================================
    # 課題：GridSearchCVでチューニングした最適パラメータを取得し，改めて外側ループのfoldに対して
    # SVMを適合させて，学習データおよびテストデータに対するaccuracyを算出し，acc_trn_list，
    # acc_tst_listに追加する．

    [YOUR CODE HERE]
    #==========================================================

print('Average of best parameters: {:.3f}'.format(np.mean(turned_params)))
print('Average accuracy on training data: {:.3f}'.format(np.mean(acc_trn_list)))
print('Average accuracy on test data: {:.3f}'.format(np.mean(acc_tst_list)))
