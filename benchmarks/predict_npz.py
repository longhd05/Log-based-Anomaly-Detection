import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def to_label(value):
    text = str(value).strip().lower()
    if text in {"anomaly", "abnormal", "1", "true", "yes"}:
        return 1
    if text in {"normal", "0", "false", "no"}:
        return 0
    return int(float(text))


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_extractor"]


def load_data(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    sequences = [
        [str(x) for x in row] if isinstance(row, (list, tuple, np.ndarray)) else [str(row)]
        for row in data["x_data"]
    ]

    if "block_ids" in data.files:
        block_ids = [str(x) for x in data["block_ids"]]
    else:
        block_ids = [f"sample_{i:06d}" for i in range(len(sequences))]

    y_true = None
    if "y_data" in data.files:
        y_true = np.array([to_label(x) for x in data["y_data"]], dtype=int)

    return block_ids, sequences, y_true


def main():
    input_path = ROOT / "data" / "HDFS" / "HDFS_115k.npz"
    model_path = ROOT / "output" / "DecisionTree" / "DecisionTree.pkl"
    output_path = ROOT / "benchmarks" / "prediction_decisiontree_npz.csv"

    model, feature_extractor = load_model(model_path)
    block_ids, sequences, y_true = load_data(input_path)

    X = feature_extractor.transform(np.array(sequences, dtype=object))
    clf = getattr(model, "classifier", model)
    y_pred = np.asarray(clf.predict(X)).astype(int)

    result = pd.DataFrame({
        "BlockId": block_ids,
        "pred": y_pred,
        "label": np.where(y_pred == 1, "anomaly", "normal"),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Samples  : {len(result)}")
    print(f"Anomaly  : {(result['pred'] == 1).sum()}")
    print(f"Normal   : {(result['pred'] == 0).sum()}")
    print(f"Saved to : {output_path}")

    if y_true is not None:
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.6f}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.6f}")
        print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.6f}")
        print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.6f}")


if __name__ == "__main__":
    main()