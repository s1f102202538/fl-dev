# Flower Federated Learning Framework

## 概要

このディレクトリには、Flower AIフレームワークを使用した連合学習の実装が含まれています。

## アーキテクチャ

```
flower/
├── fed/                          # 連合学習の核となる実装
│   ├── algorithms/               # 学習アルゴリズム
│   ├── communication/           # 通信層の実装
│   │   ├── client/             # クライアントアプリケーション
│   │   └── server/             # サーバーアプリケーション
│   ├── data/                   # データ処理・ローダー
│   ├── models/                 # ニューラルネットワークモデル
│   ├── task/                   # タスク定義
│   └── util/                   # ユーティリティ関数
├── docs/                       # ドキュメント
└── scripts/                    # スクリプト
```

## 実験実行

### 基本実行
```bash
cd ~/fl-dev/flower
flwr run .
```

### 実験の設定

`flower/pyproject.toml` で設定
