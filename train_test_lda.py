from datetime import datetime
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from settings import FEEDBACK_WINDOW_SIZE, CONTINUOUS_TESTING_INTERVAL, CONTINUOUS_TEST_BATCH_SIZE, \
    PROJECT_RESULTS_FOLDER, EXPERIMENT_NAME
from src.data.build_dataset import build_dataset, build_continuous_dataset
from src.data.util import save_file_pickle
from src.evaluation.evaluate import build_evaluation_metrics
from src.util.dataclasses import EpochedDataSet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from itertools import islice

from src.util.util import milliseconds_to_samples


def continuous_generator(test_set):
    window_size = milliseconds_to_samples(FEEDBACK_WINDOW_SIZE)
    half_interval = int(CONTINUOUS_TESTING_INTERVAL / 2)

    for x, y in zip(test_set[0], test_set[1]):
        indices = list(range(0, len(y[4]) - window_size, CONTINUOUS_TESTING_INTERVAL))
        for i in indices:
            lower_half = max(i - half_interval, 0)
            upper_half = i + half_interval
            highest_label_in_interval = max(y[4][lower_half: upper_half])
            yield x[:, i: i + window_size].reshape(-1), y[:4] + [highest_label_in_interval]
    return


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        x = [e[0] for e in batch]
        y = [e[1] for e in batch]
        if not batch:
            return
        yield x, y


def train_lda(dataset_file_path: Optional[str] = None, dataset: Optional[EpochedDataSet] = None,
              continous_dataset_path=None):
    train_set, val_set, test_set = build_dataset(dataset_file_path, dataset)

    x_train = [sample[0].reshape(-1) for sample in train_set]
    y_train = [sample[1][4] for sample in train_set]

    x_val = [sample[0].reshape(-1) for sample in val_set]
    y_val = [sample[1][4] for sample in val_set]


    pre_pca = PCA()
    pre_pca.fit(x_train, y_train)
    n_pca_features = sum(pre_pca.explained_variance_ratio_.cumsum() <= 0.99)
    print(f"PCA would need {n_pca_features}"
          f" features to explain 99% of variance")

    pipeline = Pipeline([
        ("pca", PCA(n_components=n_pca_features)),
        ("lda", LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])

    print("Training LDA...")
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_val)
    print("Validation report:")
    print(classification_report(y_val, y_pred))

    _, _, test_set = build_continuous_dataset(continous_dataset_path)

    results = [(pipeline.predict_proba(sample[0]), sample[1]) for sample in
               tqdm(batched(continuous_generator(test_set), CONTINUOUS_TEST_BATCH_SIZE),
                    total=561152 / CONTINUOUS_TEST_BATCH_SIZE)]
    y_predicted = np.concatenate([result[0][:, 1] for result in results])
    y_variance = np.zeros_like(y_predicted)
    y_true = np.array([result[4] for batch in results for result in batch[1]])
    y_in_distribution = np.ones_like(y_true)
    y_in_distribution[y_true == -1] = 0
    y_true[y_true == -1] = 1
    y_subj_idx = np.array([result[0] for batch in results for result in batch[1]])
    metrics = build_evaluation_metrics(torch.from_numpy(y_true),
                                       torch.from_numpy(y_predicted),
                                       torch.from_numpy(y_variance),
                                       torch.from_numpy(y_in_distribution),
                                       torch.from_numpy(y_subj_idx))
    save_file_pickle(metrics, PROJECT_RESULTS_FOLDER +
                     f"metrics_{EXPERIMENT_NAME}_continuous_{datetime.now().strftime('[%Y-%m-%d,%H:%M]')}.pkl")
