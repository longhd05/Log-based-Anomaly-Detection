import sys
import time
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from loglizer.models import *
from loglizer import dataloader, preprocessing

# =========================
# Config
# =========================
run_models = [
    'PCA',
    'InvariantsMiner',
    'LogClustering',
    'IsolationForest',
    'LR',
    'SVM',
    'DecisionTree'
]

STRUCT_LOG = ROOT_DIR / 'data' / 'HDFS' / 'HDFS.npz'
OUTPUT_DIR = ROOT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / 'benchmark_result_by_size.csv'
TIME_CSV = OUTPUT_DIR / 'benchmark_time_by_size.csv'

# Các mốc kích thước dữ liệu
# Bạn có thể đổi thành [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
SIZE_RATIOS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

# Nếu muốn bám theo thứ tự thời gian/phát sinh dữ liệu hơn,
# dùng sequential. Nếu chỉ benchmark chung thì uniform.
SPLIT_TYPE = 'sequential'
TRAIN_RATIO = 0.8


def slice_prefix(x, y, n):
    n = max(1, min(n, len(x)))
    return x[:n], y[:n]


def build_model_and_features(model_name, x_tr, y_train):
    if model_name == 'PCA':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(
            x_tr,
            term_weighting='tf-idf',
            normalization='zero-mean'
        )
        model = PCA()
        model.fit(x_train)

    elif model_name == 'InvariantsMiner':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr)
        model = InvariantsMiner(epsilon=0.5)
        model.fit(x_train)

    elif model_name == 'LogClustering':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(
            x_tr,
            term_weighting='tf-idf'
        )
        model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
        model.fit(x_train[y_train == 0, :])

    elif model_name == 'IsolationForest':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr)
        model = IsolationForest(
            random_state=2019,
            max_samples=0.9999,
            contamination=0.03,
            n_jobs=4
        )
        model.fit(x_train)

    elif model_name == 'LR':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(
            x_tr,
            term_weighting='tf-idf'
        )
        model = LR()
        model.fit(x_train, y_train)

    elif model_name == 'SVM':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(
            x_tr,
            term_weighting='tf-idf'
        )
        model = SVM()
        model.fit(x_train, y_train)

    elif model_name == 'DecisionTree':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(
            x_tr,
            term_weighting='tf-idf'
        )
        model = DecisionTree()
        model.fit(x_train, y_train)

    else:
        raise ValueError(f'Unknown model: {model_name}')

    return feature_extractor, x_train, model


if __name__ == '__main__':
    if not STRUCT_LOG.exists():
        raise FileNotFoundError(f'Khong tim thay file: {STRUCT_LOG}')

    print(f'Using dataset: {STRUCT_LOG}')

    # Load full dataset 1 lần
    (x_tr_full, y_train_full), (x_te_full, y_test_full) = dataloader.load_HDFS(
        str(STRUCT_LOG),
        window='session',
        train_ratio=TRAIN_RATIO,
        split_type=SPLIT_TYPE
    )

    print(f'Total train sessions: {len(x_tr_full)}')
    print(f'Total test sessions : {len(x_te_full)}')

    benchmark_results = []
    benchmark_times = []

    for ratio in SIZE_RATIOS:
        n_train = max(1, int(len(x_tr_full) * ratio))
        n_test = max(1, int(len(x_te_full) * ratio))

        x_tr, y_train = slice_prefix(x_tr_full, y_train_full, n_train)
        x_te, y_test = slice_prefix(x_te_full, y_test_full, n_test)

        print('=' * 100)
        print(f'Ratio = {ratio:.2f} | Train sessions = {len(x_tr)} | Test sessions = {len(x_te)}')

        for _model in run_models:
            print('-' * 100)
            print(f'Evaluating {_model} at ratio={ratio:.2f}')

            # =========================
            # Train time
            # =========================
            train_start = time.perf_counter()

            feature_extractor, x_train, model = build_model_and_features(_model, x_tr, y_train)

            train_end = time.perf_counter()
            train_time = train_end - train_start

            # =========================
            # Test time
            # =========================
            test_start = time.perf_counter()

            x_test = feature_extractor.transform(x_te)

            train_precision, train_recall, train_f1 = model.evaluate(x_train, y_train)
            test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test)

            test_end = time.perf_counter()
            test_time = test_end - test_start

            total_time = train_time + test_time

            # Lưu bảng accuracy theo size
            benchmark_results.append([
                _model,
                ratio,
                len(x_tr),
                len(x_te),
                'train',
                train_precision,
                train_recall,
                train_f1
            ])
            benchmark_results.append([
                _model,
                ratio,
                len(x_tr),
                len(x_te),
                'test',
                test_precision,
                test_recall,
                test_f1
            ])

            # Lưu bảng time theo size
            benchmark_times.append([
                _model,
                ratio,
                len(x_tr),
                len(x_te),
                train_time,
                test_time,
                total_time
            ])

            print(f'Train time: {train_time:.6f} sec')
            print(f'Test time : {test_time:.6f} sec')
            print(f'Total time: {total_time:.6f} sec')

    pd.DataFrame(
        benchmark_results,
        columns=[
            'Model',
            'SizeRatio',
            'NumTrainSessions',
            'NumTestSessions',
            'Split',
            'Precision',
            'Recall',
            'F1'
        ]
    ).to_csv(RESULT_CSV, index=False)

    pd.DataFrame(
        benchmark_times,
        columns=[
            'Model',
            'SizeRatio',
            'NumTrainSessions',
            'NumTestSessions',
            'TrainTimeSec',
            'TestTimeSec',
            'TotalTimeSec'
        ]
    ).to_csv(TIME_CSV, index=False)

    print(f'\n[OK] Saved: {RESULT_CSV}')
    print(f'[OK] Saved: {TIME_CSV}')