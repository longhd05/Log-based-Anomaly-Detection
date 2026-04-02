import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_time.csv")

label_map = {
    "IsolationForest": "IsolationForest",
    "DecisionTree": "DecisionTree",
    "InvariantsMiner": "InvariantsMiner",
    "LogClustering": "LogClustering",
}
df["ModelLabel"] = df["Model"].replace(label_map)

plot_df = df.copy()

group_gap = 0.82   # nhỏ hơn 1 -> các model gần nhau hơn
x = np.arange(len(plot_df)) * group_gap
width = 0.26       # tăng nhẹ độ rộng cột

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width, plot_df["TrainTimeSec"], width, label="Time train")
bars2 = ax.bar(x,         plot_df["TestTimeSec"],  width, label="Time test")
bars3 = ax.bar(x + width, plot_df["TotalTimeSec"], width, label="Total time")

ax.set_title(" ")
ax.set_ylabel("Time (seconds)")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["ModelLabel"], rotation=0, ha="center")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# bỏ khoảng trắng thừa ở 2 bên
ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

for container in [bars1, bars2, bars3]:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

plt.tight_layout()
plt.savefig("gop_3_thoi_gian.png", dpi=200, bbox_inches="tight")
plt.show()