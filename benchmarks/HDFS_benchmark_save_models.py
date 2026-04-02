import sys
import pickle
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from loglizer.models import *
from loglizer import dataloader, preprocessing

run_models = [
    'PCA',
    'InvariantsMiner',
    'LogClustering',
    'IsolationForest',
    'LR',
    'SVM',
    'DecisionTree'
]

# Chỉ dùng HDFS.npz
STRUCT_LOG = ROOT_DIR / 'data' / 'HDFS' / 'HDFS.npz'

# Output
OUTPUT_DIR = ROOT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / 'benchmark_model_result.csv'


def get_model_output_dir(model_name):
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_artifact(model_name, model, feature_extractor,
                  train_precision, train_recall, train_f1,
                  test_precision, test_recall, test_f1):
    artifact = {
        'model_name': model_name,
        'model': model,
        'feature_extractor': feature_extractor,
        'metrics': {
            'train': {
                'precision': float(train_precision),
                'recall': float(train_recall),
                'f1': float(train_f1)
            },
            'test': {
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1': float(test_f1)
            }
        },
        'data_source': str(STRUCT_LOG)
    }

    model_dir = get_model_output_dir(model_name)
    model_path = model_dir / f'{model_name}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)

    print(f'[OK] Saved model artifact: {model_path}')
    return artifact


if __name__ == '__main__':
    if not STRUCT_LOG.exists():
        raise FileNotFoundError(f'Khong tim thay file: {STRUCT_LOG}')

    print(f'Using dataset: {STRUCT_LOG}')

    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(
        str(STRUCT_LOG),
        window='session',
        train_ratio=0.8,
        split_type='uniform'
    )

    benchmark_results = []
    best_artifact = None
    best_test_f1 = -1.0

    for _model in run_models:
        print('=' * 80)
        print('Evaluating {} on HDFS:'.format(_model))

        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(
                x_tr,
                term_weighting='tf-idf',
                normalization='zero-mean'
            )
            model = PCA()
            model.fit(x_train)

        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(
                x_tr,
                term_weighting='tf-idf'
            )
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train[y_train == 0, :])

        elif _model == 'IsolationForest':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = IsolationForest(
                random_state=2019,
                max_samples=0.9999,
                contamination=0.03,
                n_jobs=4
            )
            model.fit(x_train)

        elif _model == 'LR':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(
                x_tr,
                term_weighting='tf-idf'
            )
            model = LR()
            model.fit(x_train, y_train)

        elif _model == 'SVM':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(
                x_tr,
                term_weighting='tf-idf'
            )
            model = SVM()
            model.fit(x_train, y_train)

        elif _model == 'DecisionTree':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(
                x_tr,
                term_weighting='tf-idf'
            )
            model = DecisionTree()
            model.fit(x_train, y_train)

        else:
            raise ValueError(f'Unknown model: {_model}')

        x_test = feature_extractor.transform(x_te)

        print('Train accuracy:')
        train_precision, train_recall, train_f1 = model.evaluate(x_train, y_train)
        benchmark_results.append([_model + '-train', train_precision, train_recall, train_f1])

        print('Test accuracy:')
        test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test)
        benchmark_results.append([_model + '-test', test_precision, test_recall, test_f1])

        artifact = save_artifact(
            model_name=_model,
            model=model,
            feature_extractor=feature_extractor,
            train_precision=train_precision,
            train_recall=train_recall,
            train_f1=train_f1,
            test_precision=test_precision,
            test_recall=test_recall,
            test_f1=test_f1
        )

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_artifact = artifact

    # Lưu CSV ở output/
    df = pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1'])
    df.to_csv(RESULT_CSV, index=False)
    print(f'[OK] Saved benchmark result: {RESULT_CSV}')

    # Lưu best model vào đúng folder của thuật toán thắng
    if best_artifact is not None:
        best_model_name = best_artifact['model_name']
        best_model_dir = get_model_output_dir(best_model_name)
        best_model_path = best_model_dir / f'best_model_{best_model_name}.pkl'

        with open(best_model_path, 'wb') as f:
            pickle.dump(best_artifact, f)

        print(f'[OK] Saved best model: {best_model_path}')
        print(f'[OK] Best model name: {best_model_name}')
        print(f'[OK] Best test F1: {best_test_f1:.6f}')