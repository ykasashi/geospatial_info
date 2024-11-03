# Load in our libraries
import pandas as pd
import numpy as np
from datetime import datetime
import math
# import re
# import random as random
# import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# # Going to use these 5 base models for the stacking
from sklearn.model_selection import KFold
from itertools import combinations
# from sklearn.metrics import f1_score

# machine learning
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 評価関数
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PATH = 'C:/Users/y.kasashima/Desktop/signate/20241016/'

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

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
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
        
    # 決定係数 (R^2スコア)
    r2 = r2_score(test_y, pred_y)
    
    # 結果の表示
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Sqrt Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")
    
    # 結果を辞書形式で返す
    return {
        "MAE": mae,
        "R2": r2,
        "RMSE": rmse
    }

# yyyymm形式の数値をdatetime形式に変換する関数を定義
def yyyymm_to_datetime(yyyymm_num):
    if math.isnan(yyyymm_num):  # NaNをチェック
        return None  # NaNの場合はNoneを返す
    yyyymm_int = int(yyyymm_num)  # 小数点以下を削除して整数に変換
    yyyymm_str = str(yyyymm_int)  # 数値を文字列に変換
    return datetime.strptime(yyyymm_str, "%Y%m")

def create_interaction_features(df, columns=None):
    """
    特定のカラムを除外して相互作用カラムを作成し、最後に除外したカラムを戻す関数
    
    パラメータ:
    df: pandas DataFrame
        元のデータフレーム
    target_column: str
        除外する目的変数の列名（例: 'money_room'）
    columns: list, optional
        掛け合わせたい列のリスト。デフォルトはNoneで、数値データ型の列が自動的に選択される。
    
    戻り値:
    pandas DataFrame: 元のデータフレームに新しい相互作用カラムを追加したもの
    """
    
    col_lists = []

    
    # 掛け合わせる列が指定されていない場合、数値データの列を自動選択
    if columns is None:
        columns = df.select_dtypes(include='number').columns
    
    # 2つずつの組み合わせを作成
    for col1, col2 in combinations(columns, 2):
        new_col_name = f"{col1}_x_{col2}"
        col_lists.append(new_col_name)
        df[new_col_name] = df[col1] * df[col2]
    
    return df,col_lists

# 入力
train = pd.read_csv(PATH + "train.csv")
test  = pd.read_csv(PATH + "test.csv")

# train = reduce_mem_usage(train)
# test = reduce_mem_usage(test)

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
kake_cols = ['unit_area', 'year_built_num', 'room_count', 'madori_number_all', 'flg_new', 'walk_distance1', 'money_kyoueki', 'angle_4.0','angle_5.0', 'angle_6.0', 'convenience_distance', 'super_distance', 'hospital_distance', 'room_kaisuu', 'bs_1.0', 'bs_2.0', 'bs_3.0', 'land_chisei', 'money_rimawari_now']

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

# 掛け合わせによる新しい特徴量を生成
train,train_cols = create_interaction_features(train[cols], kake_cols)
test,test_cols = create_interaction_features(test, kake_cols)
cols.extend(train_cols)

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


SEED = 0 # for reproducibility
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits= NFOLDS, shuffle=True, random_state=SEED)

# Put in our parameters for said classifiers
# Random Forest parameters
param = {
    # 二値分類問題
    'objective': 'binary:logistic',  
} 

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = Y_train.values
x_train = X_train.values # Creates an array of the train data
x_test = X_test.values # Creats an array of the test data

# 交差検証法
label = train["money_room"]
cv_train_x, cv_test_x, cv_train_y, cv_test_y = train_test_split(X_train, label, train_size = 0.8 ,test_size = 0.2, shuffle = True, random_state = 0)

# Create our OOF train and test predictions. These base results will be used as new features
print("XGBoost is Start")

reg = xgb.XGBRegressor(#目的関数の指定 初期値も二乗誤差です
                       objective='reg:squarederror',
                       #学習のラウンド数 early_stoppingを利用するので多めに指定
                       n_estimators=50000,
                       #boosterに何を用いるか 初期値もgbtreeです
                       booster='gbtree',
                       #学習率
                       learning_rate=0.01,
                       #木の最大深さ
                       max_depth=6,
                       #シード値
                       random_state=2525)

#eval_setには検証用データを設定する
reg.fit(cv_train_x, cv_train_y,eval_set=[(cv_train_x, cv_train_y),(cv_test_x, cv_test_y)])

# # データフレームの行を直接指定する
# good_data= cv_test_x.iloc[:1, :]

#予測実行
predY = reg.predict(cv_test_x)

print(accurate(cv_test_y,predY))
print("XGBoost is complete")
print("Training is complete")
