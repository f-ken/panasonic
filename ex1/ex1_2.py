#### 機械学習基礎講座第1回演習サンプルプログラム ex1_2.py
#### Programmed by Wu Hongle, 監修　福井健一
#### Last update: 2016/09/09

#### 機械学習の基本的なプロセス
#### k近傍法による分類と交差検証による評価
#### Python機械学習本：6.6.2節 (pp.168-171), Irisデータについては p. 9

from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import StratifiedKFold
np.set_printoptions(threshold=np.nan)

# K近傍法の近傍数パラメータ k
n_neighbors = 5
# Iris データセットをロード 
iris = datasets.load_iris()
# 使用する特徴の次元を(Irisの場合は0,1,2,3から)2つ指定．d1とd2は異なる次元を指定すること
d1 = 0
d2 = 1
# d1,d2列目の特徴量を使用 
X = iris.data[:, [d1, d2]]
# クラスラベルを取得
y = iris.target

# k近傍法のインスタンスを生成               
knn = KNeighborsClassifier(n_neighbors)    

# ====================== 課題(c) ====================================
# 関数StratifiedKFold()を使用し，交差検証用のトレーニング，テストデータの組を用意．
# 層化（stratified）交差検証とは，クラスのサンプル比をトレーニングとテストデータで保つようにした交差検証法である．
# 分割数は3,5,10あたりが良く使用される．適当な分割数を指定する．
kfold = [ YOUR CODE HERE ]

# ===================================================================

scores = []
precisions = []
recalls = []
fscores = []

# イテレータのインデックスと要素をループ処理
for k, (train, test) in enumerate(kfold):
    # z標準化:
    sc = StandardScaler()
    sc.fit(X[train])
    X_train_std = sc.transform(X[train])
    X_test_std = sc.transform(X[test])
    
    # モデル生成
    knn.fit(X_train_std, y[train])
    
    # ====================== 課題(d) =====================================    
    #　クラスKNeighborsClassifierの関数score()を使用してテストデータの正答率（accuracy）を算出し，変数scoreに格納．
    #　テストデータはz標準化した値を用いること
    score = [ YOUR CODE HERE ]
    # 関数precision_recall_fscore_support()を使用してテストデータの精度(precision)，再現率(recall)とF値を算出し，配列fscoreに格納
    # ヒント：KNeighborsClassifier.predict()を使用する
    fscore = [ YOUR CODE HERE ]
    
    # ===================================================================

    #テストデータのインデックスなどを出力 
    print('Fold: %s, test data index: %s, Acc: %.3f' % (k+1, test, score))
    print('Class 0 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore[0][0], fscore[1][0], fscore[2][0]))
    print('Class 1 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore[0][1], fscore[1][1], fscore[2][1]))
    print('Class 2 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore[0][2], fscore[1][2], fscore[2][2]))
    print()
    
    scores.append(score)
    precisions.append(fscore[0])
    recalls.append(fscore[1])
    fscores.append(fscore[2])

#平均値を出力    
print('CV average accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('CV average precision: %.3f +/- %.3f' % (np.mean(precisions), np.std(precisions)))
print('CV average recall: %.3f +/- %.3f' % (np.mean(recalls), np.std(recalls)))
print('CV average fscore: %.3f +/- %.3f' % (np.mean(fscores), np.std(fscores)))


