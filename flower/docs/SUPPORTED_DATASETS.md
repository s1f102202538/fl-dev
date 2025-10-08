# FederatedDatasetが使用できるデータセット

FlowerのFederatedDatasetクラスは、**Hugging Face Hubで利用可能なすべてのデータセット**を使用することができます。これは、Flower Datasetsが内部的にHugging Faceの`datasets`ライブラリを使用しているためです。

## 主な特徴

- **即座に利用可能**: Hugging Face Hubに掲載されているデータセットは、Flowerチームの承認や追加のコードなしに即座に使用できます
- **自動ダウンロード**: データセットは初回使用時に自動的にダウンロードされます
- **多様なフォーマット対応**: PyTorch、TensorFlow、NumPy、Pandas、JAX、Arrowに対応

## 推奨データセット（Flower Datasetsドキュメントより）

### 画像データセット

| データセット名 | サンプル数 | 画像サイズ | 説明 |
|---|---|---|---|
| `ylecun/mnist` | train 60k; test 10k | 28x28 | 手書き数字認識 |
| `uoft-cs/cifar10` | train 50k; test 10k | 32x32x3 | 10クラス物体認識 |
| `uoft-cs/cifar100` | train 50k; test 10k | 32x32x3 | 100クラス物体認識 |
| `zalando-datasets/fashion_mnist` | train 60k; test 10k | 28x28 | ファッションアイテム分類 |
| `flwrlabs/femnist` | train 814k | 28x28 | 連合学習用手書き文字 |
| `zh-plus/tiny-imagenet` | train 100k; valid 10k | 64x64x3 | 小型ImageNet |
| `flwrlabs/usps` | train 7.3k; test 2k | 16x16 | 郵便番号認識 |
| `flwrlabs/pacs` | train 10k | 227x227 | ドメイン適応データセット |
| `flwrlabs/cinic10` | train 90k; valid 90k; test 90k | 32x32x3 | CIFAR-10拡張版 |
| `flwrlabs/caltech101` | train 8.7k | varies | 101クラス物体認識 |
| `flwrlabs/office-home` | train 15.6k | varies | オフィス・ホーム環境 |
| `flwrlabs/fed-isic2019` | train 18.6k; test 4.7k | varies | 皮膚病変分類 |
| `ufldl-stanford/svhn` | train 73.3k; test 26k; extra 531k | 32x32x3 | ストリートビュー数字 |

### 音声データセット

| データセット名 | サンプル数 | バージョン/説明 |
|---|---|---|
| `google/speech_commands` | train 64.7k | v0.01 |
| `google/speech_commands` | train 105.8k | v0.02 |
| `flwrlabs/ambient-acoustic-context` | train 70.3k | 環境音響コンテキスト |
| `fixie-ai/common_voice_17_0` | varies | 14バージョン利用可能 |
| `fixie-ai/librispeech_asr` | varies | clean/other |

### 表形式データセット

| データセット名 | サンプル数 | 説明 |
|---|---|---|
| `scikit-learn/adult-census-income` | train 32.6k | 成人収入予測 |
| `jlh/uci-mushrooms` | train 8.1k | キノコ分類 |
| `scikit-learn/iris` | train 150 | アイリス分類 |
| `jiahborcn/chembl_aqsol` | train 12.9k; test 3.2k | 化学物質溶解度 |
| `jiahborcn/chembl_multiassay_activity` | train 350k; test 87.5k | 化学物質活性 |

### テキストデータセット

| データセット名 | サンプル数 | 分野 |
|---|---|---|
| `sentiment140` | train 1.6M; test 0.5k | 感情分析 |
| `google-research-datasets/mbpp` | full 974; sanitized 427 | 一般 |
| `openai/openai_humaneval` | test 164 | 一般 |
| `lukaemon/mmlu` | varies | 一般 |
| `takala/financial_phrasebank` | train 4.8k | 金融 |
| `pauri32/fiqa-2018` | train 0.9k; validation 0.1k; test 0.2k | 金融 |
| `zeroshot/twitter-financial-news-sentiment` | train 9.5k; validation 2.4k | 金融 |
| `bigbio/pubmed_qa` | train 2M; validation 11k | 医療 |
| `openlifescienceai/medmcqa` | train 183k; validation 4.3k; test 6.2k | 医療 |
| `bigbio/med_qa` | train 10.1k; test 1.3k; validation 1.3k | 医療 |

## 使用方法

1. **Hugging Face Hubでデータセットを探す**
   - https://huggingface.co/datasets にアクセス
   - 目的のデータセットを検索
   - データセット名（`owner/dataset-name`形式）をコピー

2. **FederatedDatasetで使用**
   ```python
   from flwr_datasets import FederatedDataset
   from flwr_datasets.partitioner import IidPartitioner

   # データセット名を指定（大文字小文字を区別）
   fds = FederatedDataset(
       dataset="uoft-cs/cifar10",  # 好きなデータセットを指定
       partitioners={"train": IidPartitioner(num_partitions=10)}
   )
   ```

3. **現在のデータローダーで使用**
   ```python
   from fl.flower.common.dataLoader.data_loader import DataLoaderConfig

   # 設定でデータセット名を変更
   config = DataLoaderConfig(
       dataset_name="uoft-cs/cifar10",  # または他のデータセット
       partitioner_type="dirichlet",
       alpha=0.5
   )
   ```

## 注意事項

- データセット名は**大文字小文字を区別**します
- データセットによっては初回ダウンロードに時間がかかる場合があります
- 一部のデータセットは特別な依存関係を必要とする場合があります（例：vision系には`flwr-datasets[vision]`）
- Hugging Face Hubの利用規約に従って使用してください

## データセットの確認方法

データセットの詳細情報（特徴量の名前、ラベル、サンプル数など）は、Hugging Face Hubのデータセットページで確認できます：
- https://huggingface.co/datasets/[dataset-name]

例：
- CIFAR-10: https://huggingface.co/datasets/uoft-cs/cifar10
- FashionMNIST: https://huggingface.co/datasets/zalando-datasets/fashion_mnist
