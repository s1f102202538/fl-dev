import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

y_data1 = [
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
y_data2 = [
  61.0383064516129,
  63.10483870967743,
  65.37298387096774,
  66.12903225806451,
  67.0866935483871,
  66.33064516129032,
  67.1875,
  66.07862903225806,
  66.93548387096774,
  66.68346774193549,
]
y_data3 = [
  59.657258064516135,
  65.51411290322581,
  65.91733870967742,
  66.27016129032258,
  64.79838709677419,
  65.13104838709677,
  65.60483870967742,
  65.1008064516129,
  64.375,
  64.92943548387098,
]

x_label = "Round"
y_label = "Accuracy [%]"
title = "Impact of each component on model accuracy"
legend_labels = ["Baseline + Params Share + Logit Calibration (Full)", "Baseline + Params Share (Params Share)", "MOON + Logit Share (Baseline)"]


def plot_line_graph():
  plt.figure(figsize=(10, 6))

  if y_data1:
    plt.plot(x_data, y_data1, marker="o", label=legend_labels[0], linewidth=2)
  if y_data2:
    plt.plot(x_data, y_data2, marker="s", label=legend_labels[1], linewidth=2)
  if y_data3:
    plt.plot(x_data, y_data3, marker="^", label=legend_labels[2], linewidth=2)

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)

  plt.legend()
  plt.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig("line_graph.png", dpi=300, bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  plot_line_graph()

  print("グラフが作成されました: line_graph.png")
