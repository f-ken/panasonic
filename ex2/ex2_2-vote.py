#### 機械学習基礎講座第2回演習サンプルプログラム ex2_2.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2016/09/14

#### ナイーブベイズ分類器による識別とROC,AUCによる評価
#### Python機械学習本：6.5.3節（pp. 185-187）

import numpy as np
from sklearn.preprocessing import label_binarize
from scipy.io import arff
from sklearn import cross_validation
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

# vote.arffデータの読み込み
f = open("vote.arff", "r", encoding="utf-8")
data, meta = arff.loadarff(f)

# 分類に用いる特徴量のindexを指定する
feature_index = [0, 1, 4]

# 特徴ベクトルの取得とエンコード
# sklearnのNaive Bayesは，数値データのみ受け付ける
## （参考）カテゴリ変数の主なエンコード方法
## (1) K個のクラス名を整数値(0,..,k-1)に置き換える（下記の方法）
## (2) K個のクラス名をKビットで表現する（1-of-K表現）
feature = []
for x in data:
    w = list(x)
    w.pop(-1) #クラスラベルは除外
    for i,s in enumerate(w):
        if s == b"\'n\'":
	        w[i]=0
        elif s == b"\'y\'":
	        w[i]=1
        elif s == b"?":
	        w[i]=2
    feature.append(w)

feature_array = np.array(feature)
feature_selected = feature_array[:,feature_index]

# クラスラベルの取得
class_label = [x[-1] for x in data]

# クラスラベルをバイナリ表現
class_array = np.asarray(class_label)
class_encoded = label_binarize(class_array, classes=[b'\'republican\'', b'\'democrat\'']).ravel()

# =====================================================================
# Leave-one-outクロスバリデーション    
y_train_post_list = []
y_train_list = []
y_test_post_list = []
y_test_list = []
clf = MultinomialNB()
loo = cross_validation.LeaveOneOut(len(class_encoded))
for train_index, test_index in loo:
    X_train, X_test = feature_selected[train_index], feature_selected[test_index]
    y_train, y_test = class_encoded[train_index], class_encoded[test_index]
    
    # =====================================================================
    # ナイーブベイズ分類器のインスタンスを生成し，学習データに適合させる
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    clf.fit(X_train,y_train)
    
    # =====================================================================
    # 学習データとテストデータに対する各クラスの事後確率の算出
    posterior_trn = clf.predict_proba(X_train)
    posterior_tst = clf.predict_proba(X_test)
    
    #print("True Label:", y_test)
    #print("Posterior Probability:", posterior_tst)

    # 各foldの正解と予測結果を保存
    y_train_post_list.extend(posterior_trn[:,[0]])
    y_train_list.extend(y_train)
    y_test_post_list.append(posterior_tst[0][0])
    y_test_list.extend(y_test)
        
# =====================================================================
# ROC曲線の描画とAUCの算出
fpr_trn, tpr_trn, thresholds_trn = roc_curve(y_train_list, y_train_post_list, pos_label=0)
roc_auc_trn = auc(fpr_trn, tpr_trn)
plt.plot(fpr_trn, tpr_trn, 'k--',label='ROC for training data (AUC = %0.2f)' % roc_auc_trn, lw=2, linestyle="-")

fpr_tst, tpr_tst, thresholds_tst = roc_curve(y_test_list, y_test_post_list, pos_label=0)
roc_auc_tst = auc(fpr_tst, tpr_tst)
plt.plot(fpr_tst, tpr_tst, 'k--',label='ROC for test data (AUC = %0.2f)' % roc_auc_tst, lw=2, linestyle="--")

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()
