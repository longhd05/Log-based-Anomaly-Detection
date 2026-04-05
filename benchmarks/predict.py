from pathlib import Path
from collections import OrderedDict
import argparse
import ast
import re
import sys
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT_FILE = ROOT / "data" / "HDFS" / "HDFS_100k.log_structured.csv"
# INPUT_FILE = ROOT / "data" / "HDFS" / "HDFS_115k.npz"

MODEL_FILE = ROOT / "output" / "DecisionTree" / "DecisionTree.pkl"
ALT_MODEL_FILE = ROOT / "output" / "DecisionTree" / "model_DecisionTree.pkl"
OUTPUT_FILE = ROOT / "benchmarks" / "prediction_result.csv"
LABEL_FILE = ROOT / "data" / "HDFS" / "anomaly_label.csv"
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


def load_block_ids_from_label(path: Path, expected_len: int):
    if not path.exists():
        print(f"[WARN] Label file not found: {path}")
        return None
    df_label = pd.read_csv(path)
    if "BlockId" not in df_label.columns:
        print(f"[WARN] 'BlockId' column not found in label file: {path}")
        return None
    block_ids = df_label["BlockId"].astype(str).tolist()
    if len(block_ids) < expected_len:
        print(f"[WARN] Label file too short: {len(block_ids)} < {expected_len}. Fallback to sample_xxx.")
        return None
    if len(block_ids) > expected_len:
        print(
            f"[INFO] Label file longer than x_data ({len(block_ids)} > {expected_len}). "
            f"Using first {expected_len} BlockId values."
        )
    return block_ids[:expected_len]


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)

    seqs = [parse_seq(x) for x in data["x_data"]]
    if "block_ids" in data.files:
        block_ids = [str(x) for x in data["block_ids"]]
        print("[INFO] BlockId source: npz['block_ids']")
    else:
        block_ids = load_block_ids_from_label(LABEL_FILE, len(seqs))
        if block_ids is not None:
            print(f"[INFO] BlockId source: {LABEL_FILE}")
        else:
            block_ids = [f"sample_{i:06d}" for i in range(len(seqs))]
            print("[WARN] BlockId source fallback: sample_xxx")

    df = pd.DataFrame({"BlockId": block_ids, "Sequence": seqs})
    return df, block_ids, seqs, "npz"


def load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    if path.suffix.lower() == ".npz":
        return load_npz(path)

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
def parse_args():
    parser = argparse.ArgumentParser(
        description="Dự đoán bất thường từ file log (CSV hoặc NPZ) sử dụng model đã huấn luyện."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Đường dẫn đến file log cần Detect (hỗ trợ .csv và .npz). "
            "Nếu không truyền, dùng file mặc định: data/HDFS/HDFS_100k.log_structured.csv"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = Path(args.file) if args.file else INPUT_FILE

    if not input_file.exists():
        print(f"[ERROR] File không tồn tại: {input_file}")
        sys.exit(1)

    print(f"[INFO] Input file: {input_file}")

    model_path = resolve_model_path()
    bundle = load_bundle(model_path)
    model = bundle["model"]
    fe = bundle["feature_extractor"]

    model_name = str(bundle.get("model_name", "")).lower()

    clf = get_classifier(model)

    df_raw, block_ids, seqs, source_type = load_input(input_file)
    print("\n===== RESULT PREVIEW =====")
    print("Num samples     :", len(block_ids))

    X, diag = build_features(seqs, fe)

    print("Unknown events  :", int(diag["unknown_event_count"].sum()))

    y_pred, y_score = predict(model, X, SCORE_THRESHOLD)

    result = pd.DataFrame({
        "BlockId": block_ids,
        "pred": y_pred.astype(int),
        "label": np.where(y_pred.astype(int) == 1, "anomaly", "normal"),
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("Anomaly count:", int((result["pred"] == 1).sum()))
    print("Normal count :", int((result["pred"] == 0).sum()))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()