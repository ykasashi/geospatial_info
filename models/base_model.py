# Load in our libraries
import pandas as pd
import numpy as np
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


# 入力
train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")
full_data = [train, test]

# room_count : 間取部屋数
# madori_number_all : 間取部屋数(代表)
# flg_new : 新築・未入居フラグ
# walk_distance1 : 徒歩距離1
cols = ['room_count', 'madori_number_all', 'flg_new', 'walk_distance1']
#df_train = train[cols].copy()
X_train = train[cols].copy()
Y_train = train["money_room"]
X_test  = test[cols].copy()

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