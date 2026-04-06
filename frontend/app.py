import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"

for p in [str(ROOT), str(BENCHMARKS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import (  # noqa: E402
    build_features,
    load_bundle,
    load_input,
    predict as run_predict,
    resolve_model_path,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

ALLOWED_EXTENSIONS = {"csv", "npz"}
MAX_TABLE_ROWS = 2000


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "Không có file nào được upload."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Chưa chọn file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Chỉ hỗ trợ file .csv và .npz."}), 400

    # Use a hardcoded suffix derived from validated extension to avoid path injection
    ext = file.filename.rsplit(".", 1)[1].lower()
    suffix = ".csv" if ext == "csv" else ".npz"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        # Load model bundle
        model_path = resolve_model_path()
        bundle = load_bundle(model_path)
        model = bundle["model"]
        fe = bundle["feature_extractor"]

        # Parse input file
        parsed = load_input(tmp_path)
        if parsed is None:
            return jsonify(
                {"error": "Không thể đọc file. Hãy kiểm tra định dạng CSV/NPZ."}
            ), 400

        _df_raw, block_ids, seqs, source_type = parsed

        # Feature extraction & prediction
        X, _diag = build_features(seqs, fe)
        y_pred, y_score = run_predict(model, X)

        pred_int = y_pred.astype(int)
        total = len(pred_int)
        anomaly_count = int((pred_int == 1).sum())
        normal_count = total - anomaly_count

        # Build row data for the table (cap at MAX_TABLE_ROWS)
        truncated = total > MAX_TABLE_ROWS
        n_display = min(total, MAX_TABLE_ROWS)
        rows = []
        for i in range(n_display):
            rows.append(
                {
                    "row": i + 1,
                    "block_id": str(block_ids[i]),
                    "score": round(float(y_score[i]), 4),
                    "label": "anomaly" if pred_int[i] == 1 else "normal",
                    "is_anomaly": bool(pred_int[i] == 1),
                }
            )

        return jsonify(
            {
                "total": total,
                "anomaly_count": anomaly_count,
                "normal_count": normal_count,
                "source_type": source_type,
                "rows": rows,
                "truncated": truncated,
            }
        )

    except Exception as exc:
        app.logger.error("Detection error: %s", exc, exc_info=True)
        return jsonify({"error": "Lỗi xử lý: " + type(exc).__name__}), 500

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=5000)
