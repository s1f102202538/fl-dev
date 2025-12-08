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
y_data1 = [8.7, 26.099999999999998, 43.5, 60.89999999999999, 78.3, 95.69999999999999, 113.1, 130.5, 147.89999999999998, 165.29999999999998]
y_data2 = [2.4579999999999997, 13.616, 24.773999999999997, 35.931999999999995, 47.089999999999996, 58.248, 69.40599999999999, 80.564, 91.722, 102.88]
y_data3 = [
  2.4579999999999997,
  7.373999999999999,
  12.29,
  17.206,
  22.121999999999996,
  27.037999999999997,
  31.953999999999997,
  36.87,
  41.785999999999994,
  46.702,
]

# ラベル設定
x_label = "Round"
y_label = "Communication cost [MB]"
title = "Communication cost"
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
