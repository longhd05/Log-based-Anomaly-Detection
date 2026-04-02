from pathlib import Path
from collections import OrderedDict
import ast
import re
import sys
import pickle
import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# INPUT_FILE = ROOT / "data" / "HDFS" / "HDFS_100k.log_structured.csv"
INPUT_FILE = ROOT / "data" / "HDFS" / "HDFS_115k.npz"

MODEL_FILE = ROOT / "output" / "DecisionTree" / "DecisionTree.pkl"
ALT_MODEL_FILE = ROOT / "output" / "DecisionTree" / "model_DecisionTree.pkl"
OUTPUT_FILE = ROOT / "benchmarks" / "prediction_result.csv"
SCORE_THRESHOLD = 0.5


def resolve_model_path():
    if MODEL_FILE.exists():
        return MODEL_FILE
    if ALT_MODEL_FILE.exists():
        return ALT_MODEL_FILE
    raise FileNotFoundError(f"Không tìm thấy model:\n- {MODEL_FILE}\n- {ALT_MODEL_FILE}")


def load_bundle(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# =========================
# INPUT PARSING
# =========================
def parse_seq(x):
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, np.ndarray):
        return [str(i) for i in x.tolist()]
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(i) for i in obj]
    except Exception:
        pass
    return [p for p in re.split(r"[\s,]+", s) if p]


def load_structured_csv(path: Path):
    df = pd.read_csv(path)

    if "LineId" in df.columns:
        df = df.sort_values("LineId").copy()

    data = OrderedDict()
    for _, row in df.iterrows():
        blk_ids = list(dict.fromkeys(re.findall(r"(blk_-?\d+)", str(row["Content"]))))
        for blk in blk_ids:
            data.setdefault(blk, []).append(str(row["EventId"]))

    return df, list(data.keys()), list(data.values()), "structured_csv"


def load_sequence_csv(path: Path):
    df = pd.read_csv(path)
    seq_cols = ["Sequence", "sequence", "EventSequence", "x_data", "events"]
    id_cols = ["BlockId", "block_id", "BlockID", "id", "sample_id"]

    seq_col = next((c for c in seq_cols if c in df.columns), None)

    id_col = next((c for c in id_cols if c in df.columns), None)
    if id_col is None:
        df["BlockId"] = [f"sample_{i:06d}" for i in range(len(df))]
        id_col = "BlockId"

    return df, df[id_col].astype(str).tolist(), df[seq_col].apply(parse_seq).tolist(), "sequence_csv"


def load_csv(path: Path):
    cols = set(pd.read_csv(path, nrows=5).columns)
    if {"Content", "EventId"}.issubset(cols):
        return load_structured_csv(path)
    if cols.intersection({"Sequence", "sequence", "EventSequence", "x_data", "events"}):
        return load_sequence_csv(path)
    raise ValueError("CSV không đúng định dạng hỗ trợ.")


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    if "x_data" not in data.files:
        raise ValueError(f"NPZ không có x_data. Keys: {list(data.files)}")

    seqs = [parse_seq(x) for x in data["x_data"]]
    block_ids = [str(x) for x in data["block_ids"]] if "block_ids" in data.files else [
        f"sample_{i:06d}" for i in range(len(seqs))
    ]

    df = pd.DataFrame({"BlockId": block_ids, "Sequence": seqs})
    return df, block_ids, seqs, "npz"


def load_input(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy input file: {path}")
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    if path.suffix.lower() == ".npz":
        return load_npz(path)
    raise ValueError(f"Chỉ hỗ trợ .csv hoặc .npz, nhận được: {path.suffix}")


# =========================
# FEATURE + PREDICT
# =========================
def get_classifier(model):
    return getattr(model, "classifier", model)


def build_features(seqs, fe):
    X = fe.transform(np.array(seqs, dtype=object))
    vocab = set(fe.events)
    num_events = np.array([len(s) for s in seqs], dtype=int)
    unknown_count = np.array([sum(e not in vocab for e in s) for s in seqs], dtype=int)
    unknown_ratio = np.where(num_events > 0, unknown_count / num_events, 0.0)
    return X, {
        "num_events": num_events,
        "unknown_event_count": unknown_count,
        "unknown_event_ratio": unknown_ratio,
    }


def predict(model, X, threshold=0.5):
    clf = get_classifier(model)
    if hasattr(clf, "predict_proba"):
        score = np.asarray(clf.predict_proba(X))[:, 1]
        pred = (score >= threshold).astype(int)
    else:
        pred = np.asarray(clf.predict(X)).astype(int).reshape(-1)
        score = pred.astype(float)
    return pred, score


# =========================
# MAIN
# =========================
def main():
    model_path = resolve_model_path()

    print("ROOT        :", ROOT)
    print("INPUT_FILE  :", INPUT_FILE)
    print("MODEL_FILE  :", model_path)
    print("OUTPUT_FILE :", OUTPUT_FILE)

    bundle = load_bundle(model_path)
    model = bundle["model"]
    fe = bundle["feature_extractor"]

    model_name = str(bundle.get("model_name", "")).lower()
    if model_name and model_name != "decisiontree":
        raise ValueError(f"Artifact không phải DecisionTree: {bundle.get('model_name')}")

    clf = get_classifier(model)

    print("\n===== MODEL INFO =====")
    print("Loaded model       :", bundle.get("model_name", "Unknown"))
    print("Classifier type    :", type(clf).__name__)
    print("Number of features :", getattr(clf, "n_features_in_", "Unknown"))
    print("Event vocabulary   :", list(fe.events))
    print("Term weighting     :", getattr(fe, "term_weighting", None))
    print("Normalization      :", getattr(fe, "normalization", None))
    print("Use OOV feature    :", getattr(fe, "oov", None))

    df_raw, block_ids, seqs, source_type = load_input(INPUT_FILE)

    print("\n===== INPUT INFO =====")
    print("Source type     :", source_type)
    print("Num samples     :", len(block_ids))
    print("First BlockId   :", block_ids[0] if block_ids else "N/A")
    print("First sequence  :", seqs[0][:20] if seqs else [])

    X, diag = build_features(seqs, fe)

    print("\n===== FEATURE INFO =====")
    print("X shape         :", X.shape)
    print("Unknown events  :", int(diag["unknown_event_count"].sum()))

    y_pred, y_score = predict(model, X, SCORE_THRESHOLD)

    result = pd.DataFrame({
        "BlockId": block_ids,
        "pred": y_pred.astype(int),
        "label": np.where(y_pred.astype(int) == 1, "anomaly", "normal"),
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n===== RESULT PREVIEW =====")
    print(result.head(20).to_string(index=False))
    print("\nAnomaly count:", int((result["pred"] == 1).sum()))
    print("Normal count :", int((result["pred"] == 0).sum()))
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()