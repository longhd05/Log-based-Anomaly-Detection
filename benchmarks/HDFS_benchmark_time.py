import sys
import time
from pathlib import Path
sys.path.append('../')

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = ROOT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing

run_models = ['PCA', 'InvariantsMiner', 'LogClustering', 'IsolationForest', 'LR', 'SVM', 'DecisionTree']
# run_models = [
#     'DecisionTree'
# ]
struct_log = '../data/HDFS/HDFS.npz'  # The benchmark dataset

if __name__ == '__main__':
    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(
        struct_log,
        window='session',
        train_ratio=0.8,
        split_type='uniform'
    )

    benchmark_summary = []

    for _model in run_models:
        print('=' * 80)
        print('Evaluating {} on HDFS:'.format(_model))

        # =========================
        # TRAIN TIME
        # =========================
        train_start = time.perf_counter()

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
            model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

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

        train_end = time.perf_counter()
        train_time = train_end - train_start

        # =========================
        # TEST TIME
        # =========================
        test_start = time.perf_counter()

        x_test = feature_extractor.transform(x_te)

        print('Train accuracy:')
        train_precision, train_recall, train_f1 = model.evaluate(x_train, y_train)

        print('Test accuracy:')
        test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test)

        test_end = time.perf_counter()
        test_time = test_end - test_start

        total_time = train_time + test_time

        benchmark_summary.append([
            _model,
            train_precision,
            train_recall,
            train_f1,
            test_precision,
            test_recall,
            test_f1,
            train_time,
            test_time,
            total_time
        ])

        print('Train time: {:.6f} sec'.format(train_time))
        print('Test time : {:.6f} sec'.format(test_time))
        print('Total time: {:.6f} sec'.format(total_time))

        pd.DataFrame(
            benchmark_summary,
            columns=[
                'Model',
                'TrainPrecision', 'TrainRecall', 'TrainF1',
                'TestPrecision', 'TestRecall', 'TestF1',
                'TrainTimeSec', 'TestTimeSec', 'TotalTimeSec'
            ]
        ).to_csv(OUTPUT_DIR / 'benchmark_time.csv', index=False)
    

    print('\n[OK] Saved: benchmark_time.csv')