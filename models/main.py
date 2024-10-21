# Load in our libraries
import pandas as pd
import numpy as np
from datetime import datetime
import math
# import re
# import random as random
# import xgboost as xgb

# import warnings
# warnings.filterwarnings('ignore')

# # Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.model_selection import KFold
# from sklearn.metrics import f1_score

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 評価関数
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PATH = 'C:/Users/y.kasashima/Desktop/signate/20241016/'

def accurate(test_y, pred_y):
    """
    モデルの評価指標を計算して表示する関数
    
    パラメータ:
    test_y: 実際の値（正解ラベル）
    pred_y: 予測された値（モデルの出力）
    
    戻り値:
    評価指標の辞書
    """
    
    # 平均絶対誤差 (MAE)
    mae = mean_absolute_error(test_y, pred_y)
        
    # 決定係数 (R^2スコア)
    r2 = r2_score(test_y, pred_y)
    
    # 結果の表示
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")
    
    # 結果を辞書形式で返す
    return {
        "MAE": mae,
        "R2": r2
    }

# yyyymm形式の数値をdatetime形式に変換する関数を定義
def yyyymm_to_datetime(yyyymm_num):
    if math.isnan(yyyymm_num):  # NaNをチェック
        return None  # NaNの場合はNoneを返す
    yyyymm_int = int(yyyymm_num)  # 小数点以下を削除して整数に変換
    yyyymm_str = str(yyyymm_int)  # 数値を文字列に変換
    return datetime.strptime(yyyymm_str, "%Y%m")

# 入力
train = pd.read_csv(PATH + "train.csv")
test  = pd.read_csv(PATH + "test.csv")
full_data = [train, test]

# 追加済み変数
# ・unit_area : 専有面積
# ・year_built_num : 築年月数(築年月から経過月数を算出)
# ・room_count : 間取部屋数
# ・madori_number_all : 間取部屋数(代表)
# ・flg_new : 新築・未入居フラグ
# ・walk_distance1 : 徒歩距離1
# ・money_kyoueki : 共益費/管理費(代表)
# ・angle_4.0~angle_6.0 : 8方角を8種類の列として表現し、南東、南、南西の3種類
# ・convenience_distance : コンビニからの距離
# ・super_distance : スーパーからの距離
# ・hospital_distance : 総合病院からの距離
# ・room_kaisuu : 部屋階数
# ・building_structure : 建物構造
# ・land_chisei : 地勢
# ・money_rimawari_now : 現行利回り
cols = ['unit_area', 'year_built_num', 'room_count', 'madori_number_all', 'flg_new', 'walk_distance1', 'money_kyoueki', 'angle_4.0','angle_5.0', 'angle_6.0', 'convenience_distance', 'super_distance', 'hospital_distance', 'room_kaisuu', 'bs_1.0', 'bs_2.0', 'bs_3.0', 'land_chisei', 'money_rimawari_now', 'money_room']

# 前処理
# 築経過月数の算出
current_date = datetime.now()
train['year_built_dt'] = train['year_built'].apply(yyyymm_to_datetime)
test['year_built_dt'] = test['year_built'].apply(yyyymm_to_datetime)
train['year_built_num'] = train['year_built_dt'].apply(lambda x: (current_date.year - x.year) * 12 + current_date.month - x.month)
test['year_built_num'] = test['year_built_dt'].apply(lambda x: (current_date.year - x.year) * 12 + current_date.month - x.month)

# 方角をワンホットエンコーディングで8変数追加
train = pd.get_dummies(train, columns=['snapshot_window_angle'], prefix='angle')
test = pd.get_dummies(test, columns=['snapshot_window_angle'], prefix='angle')

# 建物構造をワンホットエンコーディングで8変数追加
train = pd.get_dummies(train, columns=['building_structure'], prefix='bs')
test = pd.get_dummies(test, columns=['building_structure'], prefix='bs')

# 相関行列を計算
correlation_matrix = train[cols].corr()
correlation_with_target = correlation_matrix['money_room'].abs()
selected_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()

# ターゲット変数以外の選択された特徴量を表示
selected_features.remove('money_room')
print("選択された特徴量:", selected_features)

# 分割
X_train = train[selected_features].copy()
Y_train = train["money_room"]
X_test  = test[selected_features].copy()


# 交差検証法
label = train["money_room"]
cv_train_x, cv_test_x, cv_train_y, cv_test_y = train_test_split(X_train, label, train_size = 0.8 ,test_size = 0.2, shuffle = True, random_state = 0)

SEED = 0 # for reproducibility
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits= NFOLDS, shuffle=True, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    #'n_jobs': -1,
    'n_estimators': 3,
     #'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 3,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = Y_train.values
x_train = X_train.values # Creates an array of the train data
x_test = X_test.values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
print("Random Forest is Start")
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test) # Random Forest
print(accurate(y_train,rf_oof_train))
print("Random Forest is complete")
print("Training is complete")
