# Load in our libraries
import pandas as pd
import numpy as np
from datetime import datetime
import math

import warnings
warnings.filterwarnings('ignore')

# # Going to use these 5 base models for the stacking
from sklearn.model_selection import KFold
from itertools import combinations

# machine learning
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 評価関数
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PATH = 'C:/Users/y.kasashima/Desktop/signate/20241016/'

class XGBoostTrainer:
    def __init__(self, data_path, param_json_path):
        """
        XGBoostTrainerの初期化
        """
        self.data_path = data_path
        self.xgboost_params = self.load_params(param_json_path)
        self.model = None

    def load_params(self, param_json_path):
        """
        JSONファイルからXGBoostのパラメータを読み込む関数
        """
        try:
            with open(param_json_path, 'r') as file:
                params = json.load(file)
            print("Parameters loaded from JSON file.")
            return params
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return {
                'objective': 'reg:squarederror',  # デフォルト設定
                'n_estimators': 50000,
                'learning_rate': 0.01,
                'max_depth': 6,
                'booster': 'gbtree',
                'random_state': 2525
            }
    
    def load_data(self):
        """
        データを読み込む関数
        """
        self.train_data = pd.read_csv(self.data_path + "train.csv")
        self.test_data = pd.read_csv(self.data_path + "test.csv")
        print("Data loaded successfully.")
        
    @staticmethod
    def yyyymm_to_datetime(yyyymm_num):
        """
        yyyymm形式の数値をdatetime形式に変換する関数
        """
        if math.isnan(yyyymm_num):
            return None
        yyyymm_int = int(yyyymm_num)
        yyyymm_str = str(yyyymm_int)
        return datetime.strptime(yyyymm_str, "%Y%m")
    
    @staticmethod
    def accurate(test_y, pred_y):
        """
        モデルの評価指標を計算して表示する関数
        """
        mae = mean_absolute_error(test_y, pred_y)
        rmse = np.sqrt(mean_squared_error(test_y, pred_y))
        r2 = r2_score(test_y, pred_y)
        
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R^2 Score: {r2}")
        
        return {"MAE": mae, "R2": r2, "RMSE": rmse}

    def prepare_data(self):
        """
        特徴量とターゲットの分割と、トレーニング・テストデータの分割
        """
        self.X_train = self.train_data.drop(columns=["money_room"])
        self.Y_train = self.train_data["money_room"]
        
        self.cv_train_x, self.cv_test_x, self.cv_train_y, self.cv_test_y = train_test_split(
            self.X_train, self.Y_train, train_size=0.8, test_size=0.2, shuffle=True, random_state=self.xgboost_params['random_state']
        )
        print("Data preparation completed.")
    
    def train(self):
        """
        モデルのトレーニング関数
        """
        self.model = xgb.XGBRegressor(**self.xgboost_params)
        self.model.fit(
            self.cv_train_x, self.cv_train_y,
            eval_set=[(self.cv_train_x, self.cv_train_y), (self.cv_test_x, self.cv_test_y)],
            early_stopping_rounds=100, verbose=False
        )
        print("Model training completed.")

    def evaluate(self):
        """
        テストデータに対してモデルを評価する関数
        """
        pred_y = self.model.predict(self.cv_test_x)
        return self.accurate(self.cv_test_y, pred_y)

# 使用例
data_path = 'C:/Users/y.kasashima/Desktop/signate/20241016/'
param_json_path = 'C:/Users/y.kasashima/Desktop/signate/xgboost_params.json'

trainer = XGBoostTrainer(data_path, param_json_path)
trainer.load_data()
trainer.prepare_data()
trainer.train()
evaluation_metrics = trainer.evaluate()
print(evaluation_metrics)