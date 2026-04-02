import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_model_result.csv")

# Tách tên model và loại tập dữ liệu (train/test)
parsed = df["Model"].str.extract(r"^(.*?)-(train|test)$")
df["BaseModel"] = parsed[0]
df["Dataset"] = parsed[1]

# Chọn các model cần vẽ
selected_models = ['PCA', 'InvariantsMiner', 'LogClustering', 'IsolationForest']
# selected_models = ['LR', 'SVM', 'DecisionTree']

metrics = ["Precision", "Recall", "F1"]

def make_chart(dataset_name, output_file, title):
    plot_df = (
        df[df["BaseModel"].isin(selected_models) & (df["Dataset"] == dataset_name)]
        .set_index("BaseModel")
        .loc[selected_models, metrics]
    )

    x = np.arange(len(selected_models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, plot_df["Precision"], width, label="Precision")
    ax.bar(x,         plot_df["Recall"],    width, label="Recall")
    ax.bar(x + width, plot_df["F1"],        width, label="F1-score")

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(selected_models, rotation=0, ha ="center")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.show()

make_chart("train", "bieu_do_train_4_models.png", " ")
make_chart("test", "bieu_do_test_4_models.png", " ")