# geospatial_info

## フォルダ構成
geospatial_info/
│
├── data/
│   ├── raw/               # 未加工のデータ（元データ）
│   ├── processed/          # 前処理後のデータ（クリーンデータ）
│   └── external/           # 外部ソースのデータ（公開データセットなど）
│
├── notebooks/              # Jupyterノートブック
│   ├── exploratory.ipynb   # 初期の探索的データ分析（EDA）
│   └── model_training.ipynb # モデルのトレーニング
│
├── scripts/                # Pythonスクリプト（再利用可能なコード）
│   ├── data_processing.py  # データ前処理関連のスクリプト
│   └── model_training.py   # モデルの学習・評価を行うスクリプト
│
├── models/                 # 保存されたモデル
│   └── best_model.pkl      # トレーニング後の最良モデル
│
├── requirements.txt        # 必要なPythonパッケージ
├── README.md               # プロジェクトの概要や使い方
├── setup.py                # パッケージの設定ファイル（必要に応じて）
└── .gitignore              # Gitで追跡しないファイルのリスト（例: .DS_Store、.env など）
