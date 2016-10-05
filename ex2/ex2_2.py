#### 機械学習基礎講座第2回演習サンプルプログラム ex2_2.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2016/09/14

#### ナイーブベイズ分類器による識別とROC,AUCによる評価
#### Python機械学習本：6.5.3節（pp. 185-187）

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer,label_binarize,LabelEncoder
from scipy.io import arff
from sklearn import cross_validation
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

# arffデータの読み込み
f = open("weather.nominal.arff", "r", encoding="utf-8")
data, meta = arff.loadarff(f)

# ラベルエンコーダの設定
le = [LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(),LabelEncoder()]
for idx,attr in enumerate(meta):
    le[idx].fit(meta._attributes[attr][1])

# 特徴ベクトルとクラスラベルの取得とエンコード
# sklearnのNaive Bayesは，数値データのみ受け付ける
## （参考）カテゴリ変数の主なエンコード方法
## (1) K個のクラス名を整数値(0,..,k-1)に置き換える（本サンプルの方法）
## (2) K個のクラス名をKビットでひとつだけ1，残り全て0で表現する（1-of-K表現）
feature = []
class_label = []
for x in data:
    w = list(x)
    class_label.append(le[-1].transform(w[-1].decode("utf-8")))
    w.pop(-1)
    for idx in range(0, len(w)):
        w[idx] = le[idx].transform(w[idx].decode("utf-8"))
    feature.append(w)

feature_array = np.array(feature)
class_array = np.array(class_label)

# =====================================================================
print("Leave-one-out Cross-validation")
y_train_post_list = []
y_train_list = []
y_test_post_list = []
y_test_list = []

loo = cross_validation.LeaveOneOut(len(class_label))
for train_index, test_index in loo:
    X_train, X_test = feature_array[train_index], feature_array[test_index]
    y_train, y_test = class_array[train_index], class_array[test_index]
    
    # =====================================================================
    # 課題(c) ナイーブベイズ分類器のインスタンスを生成し，学習データに適合させる．
    # 多項ナイーブベイズ（MultinomialNB）を使用すること．
    ## オプションのalphaは，機械学習入門p.75の等価標本サイズmと若干異なるため注意．
    ## http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
    [YOUR CODE HERE]
    
    # =====================================================================
    # 課題(d) 学習データとテストデータに対する各クラスの事後確率を算出
    posterior_trn = [YOUR CODE HERE]
    posterior_tst = [YOUR CODE HERE]
    
    print("True Label:", y_test)
    print("Posterior Probability:", posterior_tst)

    # 正解クラスと事後確率を保存
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
