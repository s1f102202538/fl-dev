"""matplotlib 折れ線グラフ作成テンプレート"""

import matplotlib.pyplot as plt

# フォント・スタイル設定
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# ===== データ入力部分 =====
# ここにデータを入力してください

# X軸データ（例: ラウンド数、エポック数など）
x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Y軸データ（複数系列）
y_data1 = [
  25.90725806451613,
  57.006048387096776,
  63.659274193548384,
  69.05241935483872,
  69.30443548387096,
  72.47983870967742,
  72.02620967741935,
  73.08467741935483,
  74.44556451612904,
  74.69758064516128,
]
y_data2 = [
  62.903225806451616,
  67.0866935483871,
  66.98588709677419,
  67.69153225806451,
  69.45564516129032,
  67.99395161290323,
  69.10282258064517,
  69.85887096774194,
  69.00201612903226,
  68.24596774193549,
]
y_data3 = [
  59.94959677419355,
  61.653225806451616,
  60.03024193548388,
  60.66532258064516,
  61.75403225806451,
  60.77620967741936,
  59.13306451612903,
  59.284274193548384,
  60.27217741935484,
  60.24193548387096,
]

# ラベル設定
x_label = "Round"
y_label = "Accuracy [%]"
title = "Model accuracy"
legend_labels = ["FedAvg", "Proposed method", "FedMD"]  # 凡例ラベル

# ===== グラフ作成部分 =====


def plot_line_graph():
  """折れ線グラフを作成"""
  plt.figure(figsize=(10, 6))

  # データ系列をプロット（必要に応じてコメントアウト/追加）
  if y_data1:
    plt.plot(x_data, y_data1, marker="o", label=legend_labels[0], linewidth=2)
  if y_data2:
    plt.plot(x_data, y_data2, marker="s", label=legend_labels[1], linewidth=2)
  if y_data3:
    plt.plot(x_data, y_data3, marker="^", label=legend_labels[2], linewidth=2)

  # 軸ラベルとタイトル
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)

  # 凡例とグリッド
  plt.legend()
  plt.grid(True, alpha=0.3)

  # レイアウト調整と保存
  plt.tight_layout()
  plt.savefig("line_graph.png", dpi=300, bbox_inches="tight")
  plt.show()


# ===== 実行部分 =====
if __name__ == "__main__":
  # データを入力後、この関数を実行
  plot_line_graph()

  print("グラフが作成されました: line_graph.png")
