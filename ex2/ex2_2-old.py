#### 機械学習基礎講座第2回演習サンプルプログラム ex2_2.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2016/09/07

#### ナイーブベイズ分類器による識別とROC,AUCによる評価
#### Python機械学習本：6.5.3節（pp. 185-187）

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer,label_binarize
from scipy.io import arff
from sklearn import cross_validation
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

# weather.nominal.arffデータの読み込み
f = open("weather.nominal.arff", "r", encoding="utf-8")
data, meta = arff.loadarff(f)

# 特徴ベクトルの取得
feature = []
for i in data:
	w = list(i)
	w.pop(-1)
	feature.append(w)
# クラスラベルの取得
class_original = [x[-1] for x in data]

# 1-of-K表現でカテゴリ変数をバイナリに変換
mlb = MultiLabelBinarizer()
feature_encoded = mlb.fit_transform(feature)
feature_names = mlb.classes_
#print("\n============ Training ============")
print("Features: ",feature_names)
print(feature_encoded)

# クラスラベルをバイナリ表現
class_encoded = label_binarize(np.array(class_original), classes=[b'no', b'yes']).ravel()
print(class_encoded)

# =====================================================================
print("Leave-one-out Cross-validation")
y_train_post_list = []
y_train_list = []
y_test_post_list = []
y_test_list = []
#clf = MultinomialNB()
clf = BernoulliNB()
loo = cross_validation.LeaveOneOut(len(class_encoded))
for train_index, test_index in loo:

    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = feature_encoded[train_index], feature_encoded[test_index]
    y_train, y_test = class_encoded[train_index], class_encoded[test_index]
    
    # =====================================================================
    # ナイーブベイズ分類器のインスタンスを生成し，学習データに適合させる
    clf.fit(X_train,y_train)
    #MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    BernoulliNB(alpha=1.0, class_prior=None, fit_prior=True)
    
    # =====================================================================
    # 学習データとテストデータに対する各クラスの事後確率の算出
    posterior_trn = clf.predict_proba(X_train)
    posterior_tst = clf.predict_proba(X_test)
    
    print("True Label:", y_test)
    print("Posterior Probability:", posterior_tst)

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
#clf = MultinomialNB()
#clf.fit(weather_encoded,playtennis)
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

#x_train_prediction_prob = clf.predict_proba(weather_encoded)
#print(x_train_prediction_prob[:, [0]])
#fpr, tpr, thresholds = roc_curve(playtennis, x_train_prediction_prob[:,[0]], pos_label=0)
#roc_auc_train = auc(fpr, tpr)
#print(roc_auc_train)