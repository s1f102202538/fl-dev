# Federated Learning Framework Documentation

## 概要

本プロジェクトは、Flower フレームワークをベースにした連合学習（Federated Learning）の実装です。クライアント間でモデルを分散学習し、サーバで集約・蒸留を行う仕組みを提供します。

## 実行環境

### 開発環境

- **OS**: Debian GNU/Linux 12 (bookworm)
- **コンテナ**: Dev Container
- **Python**: 3.12+

### 使用フレームワーク・ライブラリ

| フレームワーク/ライブラリ | バージョン | 用途 |
|------------------------|----------|------|
| **Flower** | ≥1.18.0 | 連合学習フレームワーク（シミュレーション対応） |
| **PyTorch** | 2.5.1 | 深層学習フレームワーク |
| **TorchVision** | 0.20.1 | 画像処理・データセット |
| **flwr-datasets** | ≥0.5.0 | Flower 用データセット管理 |
| **Weights & Biases** | - | 実験管理・可視化（オプション） |

## ディレクトリ構造
```
flower/
├── pyproject.toml              # プロジェクト設定（依存関係、Flower アプリ設定）
├── docs/                       # ドキュメント
│   ├── README.md              # 本ドキュメント
│   ├── LOGIT_FILTERING_EVALUATION.md
│   └── SUPPORTED_DATASETS.md
├── fed/                        # 連合学習コア実装
│   ├── algorithms/            # 学習アルゴリズム
│   │   ├── csd_based_moon.py          # CSD ベース MOON
│   │   ├── distillation.py            # 知識蒸留
│   │   ├── loca_based_distillation.py # LoCa ベース蒸留
│   │   ├── logit_calibration_moon.py  # ロジット校正 MOON
│   │   └── moon.py                    # MOON コントラスト学習
│   ├── communication/         # クライアント・サーバ通信
│   │   ├── client/           # クライアント実装
│   │   │   ├── apps/        # クライアントアプリケーション
│   │   │   │   ├── fed_avg_client.py
│   │   │   │   ├── fed_md_client.py
│   │   │   │   ├── fed_md_params_share_client.py
│   │   │   │   ├── fed_moon_client.py
│   │   │   │   ├── fed_moon_params_share_client.py
│   │   │   │   └── fed_moon_params_share_csd_client.py
│   │   │   └── client_app.py # クライアントルーティング
│   │   └── server/           # サーバ実装
│   │       ├── apps/        # サーバアプリケーション
│   │       │   ├── fed_avg_server.py
│   │       │   ├── fed_md_avg_server.py
│   │       │   ├── fed_md_distillation_model_server.py
│   │       │   ├── fed_md_distillation_model_with_training_server.py
│   │       │   ├── fed_md_params_share_server.py
│   │       │   ├── fed_md_params_share_csd_server.py
│   │       │   └── fed_md_weighted_avg_server.py
│   │       ├── strategy/    # 集約戦略
│   │       │   ├── fed_avg.py
│   │       │   ├── fed_md_avg.py
│   │       │   ├── fed_md_distillation_model.py
│   │       │   ├── fed_md_distillation_model_with_training.py
│   │       │   ├── fed_md_params_share.py
│   │       │   └── fed_md_params_share_csd.py
│   │       └── server_app.py # サーバルーティング
│   ├── data/                  # データ管理
│   │   ├── data_loader_config.py       # データローダー設定
│   │   ├── data_loader_manager.py      # データローダー管理
│   │   ├── data_transform_manager.py   # データ変換管理
│   │   └── transformed_dataset.py      # 変換済みデータセット
│   ├── models/                # モデル定義
│   │   ├── base_model.py            # 基底モデル
│   │   ├── mini_cnn.py              # Mini CNN
│   │   ├── moon_model.py            # MOON モデル
│   │   └── simple_cnn.py            # Simple CNN
│   ├── task/                  # タスク実装
│   │   └── cnn_task.py             # CNN 訓練・評価・推論
│   └── util/                  # ユーティリティ
│       ├── communication_cost.py    # 通信コスト計算
│       ├── create_model.py          # モデル生成
│       ├── create_partitioner.py    # データ分割設定
│       ├── data_loader.py           # データローダー作成
│       ├── model_util.py            # モデルユーティリティ
│       └── visualize_data.py        # データ可視化
├── outputs/                   # 実験結果出力
└── scripts/                   # スクリプト
    ├── evaluate_model_accuracy.py
    ├── plot_communication_cost.py
    ├── plot_model_accuracy.py
    ├── plot_propose_method_model_accuracy.py
    └── visualize_data_distribution.py
```

## 各モジュールの役割

### 1. fed/algorithms/ - 学習アルゴリズム

連合学習で使用する各種アルゴリズムの実装。

- **moon.py**: MOON (Model-Contrastive Federated Learning) の実装
  - グローバルモデルと前回のローカルモデルを用いたコントラスト学習
  - MoonContrastiveLearning: 対比損失の計算
  - MoonTrainer: MOON による訓練ループ

- **csd_based_moon.py**: CSD（Class Prototype Similarity Distillation）を統合した MOON
  - クラスプロトタイプとの類似度に基づく蒸留損失を追加
  - CsdBasedMoonContrastiveLearning: CSD 損失を含む対比学習
  - CsdBasedMoonTrainer: CSD + MOON の訓練ループ

- **logit_calibration_moon.py**: FedLC（ロジット校正）を統合した MOON
  - クラス不均衡に対するロジット校正を適用
  - LogitCalibrationMoonContrastiveLearning: FedLC 対応
  - LogitCalibrationMoonTrainer: 校正付き訓練

- **distillation.py**: 標準的な知識蒸留
  - サーバ側でのモデル蒸留
  - Distillation: KL divergence ベースの蒸留

- **loca_based_distillation.py**: LoCa（Logit Calibration）ベースの蒸留
  - 非正解クラスの確率を縮小し、正解クラスを強調
  - 温度スケーリングと組み合わせた蒸留

### 2. fed/communication/ - 通信層

クライアントとサーバ間の通信とデータ交換を管理。

#### client/ - クライアント側

- **client_app.py**: クライアントのルーティングハブ
  - client_name に基づいて適切なクライアントアプリを起動
  - 対応クライアント: fed-avg-client, fed-md-client, fed-md-params-share-client, fed-moon-client, fed-moon-params-share-client, fed-moon-params-share-csd-client

- **apps/**: 各クライアント実装
  - **fed_avg_client.py**: FedAvg 標準クライアント
  - **fed_md_client.py**: FedMD（知識蒸留）クライアント
  - **fed_md_params_share_client.py**: FedMD パラメータ共有 + ロジット送信クライアント
  - **fed_moon_client.py**: MOON コントラスト学習クライアント
  - **fed_moon_params_share_client.py**: MOON + パラメータ共有クライアント
  - **fed_moon_params_share_csd_client.py**: CSD-MOON クライアント

#### server/ - サーバ側

- **server_app.py**: サーバのルーティングハブ
  - server_name に基づいて適切なサーバアプリと戦略を起動
  - 対応サーバ: fed-avg-server, fed-md-avg-server, fed-md-distillation-model-server, fed-md-distillation-model-with-training-server, fed-md-params-share-server, fed-md-params-share-csd-server, fed-md-weighted-avg-server

- **strategy/**: 集約戦略
  - **fed_avg.py**: FedAvg（標準的な連合平均化）
  - **fed_md_avg.py**: FedMD 単純平均化戦略
  - **fed_md_distillation_model.py**: FedMD 蒸留モデル戦略（ロジット集約のみ）
  - **fed_md_distillation_model_with_training.py**: FedMD 蒸留モデル + 訓練戦略
  - **fed_md_params_share.py**: FedMD パラメータ配布 + ロジット集約 + 蒸留
  - **fed_md_params_share_csd.py**: FedMD CSD 統合版（クラスプロトタイプ生成）
  - **fed_md_weighted_avg.py**: FedMD 重み付き平均化戦略

- **apps/**: サーバアプリケーション
  - 各戦略を初期化し、ServerAppComponents を返す
  - 戦略ごとに対応するサーバアプリが存在

### 3. fed/data/ - データ管理

連合学習のためのデータ分割とローディング。

- **data_loader_config.py**: データローダーの設定
  - DataLoaderConfig: データセット名、バッチサイズ、分割方法などを定義
  - train_max_samples: 訓練データの最大サンプル数制限

- **data_loader_manager.py**: データローダーの管理
  - create_federated_data_loaders(): クライアント用の分割データローダー作成
  - create_public_test_data_loader(): サーバ用の公開テストデータ作成
  - IID / Non-IID 分割対応

- **data_transform_manager.py**: データ変換の管理
  - データ拡張や正規化などの変換処理を管理

- **transformed_dataset.py**: 変換済みデータセット
  - データセットに変換を適用したラッパークラス

### 4. fed/models/ - モデル定義

ニューラルネットワークモデルの実装。

- **base_model.py**: 基底クラス
  - BaseModel: 全モデルの抽象基底クラス
  - predict(), forward() メソッドを定義

- **mini_cnn.py**: Mini CNN
  - MiniCNN: 軽量な畳み込みニューラルネットワーク
  - CIFAR-10 などの小規模データセット用

- **simple_cnn.py**: Simple CNN
  - SimpleCNN: シンプルな CNN アーキテクチャ
  - 基本的な画像分類タスク用

- **moon_model.py**: MOON 用モデル
  - ModelFedCon: プロジェクションヘッド付き CNN
  - features → l1 → l2 (projection head) と features → l3 (classifier) の二つの出力

### 5. fed/task/ - タスク実装

モデルの訓練・評価・推論タスク。

- **cnn_task.py**: CNN タスクの実装
  - train(): 標準的な訓練
  - test(): テスト評価
  - inference(): ロジット生成（補正なし）
  - inference_with_label_correction(): ラベル補正付き推論
  - inference_with_loca(): LoCa 補正付き推論

### 6. fed/util/ - ユーティリティ

共通のヘルパー関数。

- **communication_cost.py**: 通信コスト計算
  - パラメータやロジットのサイズを計算
  - MB 単位での通信量測定

- **create_model.py**: モデル生成
  - モデル名に基づいたモデルインスタンスの生成
  - モデルファクトリー機能

- **create_partitioner.py**: データ分割設定
  - IID / Non-IID データ分割の設定を生成
  - Partitioner インスタンスの作成

- **data_loader.py**: データローダー作成
  - データセットからデータローダーを作成するヘルパー関数

- **model_util.py**: モデルユーティリティ
  - Base64 エンコード/デコード
  - バッチリストの変換
  - 実行ディレクトリの作成

- **visualize_data.py**: データ可視化
  - データ分布の可視化ヘルパー関数

## 実行方法

### 基本実行
```bash
flwr run .
```

### 設定のカスタマイズ

`pyproject.toml` の `[tool.flwr.app.config]` セクションで設定を変更：

```toml
[tool.flwr.app.config]
model_name = "mini-cnn"              # モデル選択（mini-cnn, simple-cnn, moon-model など）
dataset_name = "uoft-cs/cifar10"     # データセット選択
client_name = "fed-md-params-share-client"  # クライアント選択
server_name = "fed-md-params-share-server"  # サーバ選択
num_clients = 10                     # クライアント数
num_rounds = 50                      # ラウンド数
```
