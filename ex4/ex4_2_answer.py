#### 機械学習基礎講座第4回演習サンプルプログラム ex4_2.py
#### Programmed by Wu Hongle, 監修　福井健一
#### Last updated: 2016/10/04

#### 線形回帰によるHousingデータ住宅価格の推定
#### Python機械学習本：10.5節，6.4節
#### Housingデータについては，10.2節を参照

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sbs import SBS
 
# Boston Housingデータのロード       
df = load_boston()
X = df.data
y = df.target
n_of_features = len(df.feature_names)
n_of_selected_features = 5 # 特徴選択の特徴量数の指定（特徴量名の表示のみに使用）

# z標準化
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

n_of_trials = 30 # 試行回数
score_train_all = np.zeros(n_of_features) #部分集合毎の学習データに対するスコア格納用
score_test_all = np.zeros(n_of_features)  #部分集合毎のテストデータに対するスコア格納用

#==========================================================
# 本プログラムは交差検証ではなく，異なる乱数状態で複数回試行した平均を取っている
for k in range(0, n_of_trials):
	X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.3, random_state = k)

	lr = LinearRegression()
	sbs = SBS(lr, k_features=1, scoring=r2_score)
	sbs.fit(X_train, y_train)
	selected_features = list(sbs.subsets_[n_of_features - n_of_selected_features])
	print("Trial {:2d}; Best {} features: {}".format(k+1, n_of_selected_features, df.feature_names[selected_features]))

	score_train = np.array([])
	score_test = np.array([])

	#======================================================
	# 課題：SBSアルゴリズムで得られた各部分集合に対して，線形回帰モデルを適合させて
	# 学習データ，テストデータに対する決定係数を算出し，score_train，score_testに格納する．
	# ヒント：特徴の部分集合はsbs.subsets_に格納されている．
	# [YOUR CODE HERE]
	for subset in sbs.subsets_:
		lr.fit(X_train[:, subset], y_train)
		y_train_pred = lr.predict(X_train[:, subset])
		y_test_pred = lr.predict(X_test[:, subset])
	#	print('#Selected Features:%2.d R^2 train: %.3f, test: %.3f' % ( len(subset), r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred) ))
		score_train = np.append(score_train, r2_score(y_train, y_train_pred))
		score_test = np.append(score_test, r2_score(y_test, y_test_pred))
	#======================================================

	score_train_all += score_train
	score_test_all += score_test
#==========================================================

# SBSアルゴリズムで選択された特徴の部分集合と決定係数のグラフをプロット
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, score_train_all/n_of_trials, marker='o', label="Training data")
plt.plot(k_feat, score_test_all/n_of_trials, marker='x', label="Test data")
plt.ylabel('R^2 score')
plt.xlabel('Number of features')
plt.legend(loc="lower right")
plt.grid()
plt.show()
