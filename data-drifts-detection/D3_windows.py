import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.data.data_stream import DataStream
import time
import sys


# Drift Detector
def drift_detector(S, T, threshold=0.75):
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    T['in_target'] = 0  # in target set
    S['in_target'] = 1  # in source set
    ST = pd.concat([T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values
    clf = LogisticRegression(solver='liblinear')
    predictions = np.zeros(labels.shape)
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = AUC(labels, predictions)
    return auc_score > threshold


class D3:
    def __init__(self, w, rho, dim, auc):
        self.size = int(w * (1 + rho))
        self.win_data = np.zeros((self.size, dim))
        self.win_label = np.zeros(self.size)
        self.w = w
        self.rho = rho
        self.dim = dim
        self.auc = auc
        self.drift_count = 0
        self.window_index = 0

    def addInstance(self, X, y):
        if self.isEmpty():
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.window_index += 1
        else:
            print("Error: Buffer is full!")

    def isEmpty(self):
        return self.window_index < self.size

    def driftCheck(self):
        if drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size], self.auc):
            self.window_index = int(self.w * self.rho)
            self.win_data = np.roll(self.win_data, -1 * self.w, axis=0)
            self.win_label = np.roll(self.win_label, -1 * self.w, axis=0)
            self.drift_count += 1
            return True
        else:
            self.window_index = self.w
            self.win_data = np.roll(self.win_data, -1 * int(self.w * self.rho), axis=0)
            self.win_label = np.roll(self.win_label, -1 * int(self.w * self.rho), axis=0)
            return False

    def getCurrentData(self):
        return self.win_data[:self.window_index]

    def getCurrentLabels(self):
        return self.win_label[:self.window_index]


def select_data(x):
    df = pd.read_csv(x)
    scaler = MinMaxScaler()
    df.iloc[:, 0:df.shape[1] - 1] = scaler.fit_transform(df.iloc[:, 0:df.shape[1] - 1])
    return df


def check_true(y, y_hat):
    return int(y == y_hat)


def run_drift_detection(window_size, rho, auc_threshold, df):
    stream = DataStream(df)
    stream_clf = HoeffdingTreeClassifier()
    
    D3_win = D3(window_size, rho, stream.n_features, auc_threshold)
    stream_true = 0
    stream_acc = []
    
    i = 0
    start = time.time()
    X, y = stream.next_sample(int(window_size * rho))
    stream_clf.partial_fit(X, y, classes=stream.target_values)
    
    while stream.has_more_samples():
        X, y = stream.next_sample()
        if D3_win.isEmpty():
            D3_win.addInstance(X, y)
            y_hat = stream_clf.predict(X)
            stream_true += check_true(y, y_hat)
            stream_clf.partial_fit(X, y)
            stream_acc.append(stream_true / (i + 1))
        else:
            if D3_win.driftCheck():  # detected drift
                stream_clf.reset()
                stream_clf.partial_fit(D3_win.getCurrentData(), D3_win.getCurrentLabels(), classes=stream.target_values)
                y_hat = stream_clf.predict(X)
                stream_true += check_true(y, y_hat)
                stream_clf.partial_fit(X, y)
                stream_acc.append(stream_true / (i + 1))
                D3_win.addInstance(X, y)
            else:
                y_hat = stream_clf.predict(X)
                stream_true += check_true(y, y_hat)
                stream_clf.partial_fit(X, y)
                stream_acc.append(stream_true / (i + 1))
                D3_win.addInstance(X, y)
        i += 1

    elapsed_time = time.time() - start
    final_acc = stream_acc[-1] * 100
    return final_acc, elapsed_time


def plot_final_accuracy_vs_window_size(df, rho, auc_threshold):
    window_sizes = [100, 250, 500, 1000, 2500]
    final_accuracies = []

    for window_size in window_sizes:
        print(f"Running for old data size {window_size}")
        final_acc, _ = run_drift_detection(window_size, rho, auc_threshold, df)
        final_accuracies.append(final_acc)

    plt.figure()
    plt.plot(window_sizes, final_accuracies, marker='o')
    plt.xlabel("Old Data Size")
    plt.ylabel("Final Accuracy (%)")
    plt.title("Final Accuracy vs. Old Data Size")
    plt.grid(True)
    plt.show()


# Parameters
dataset_path = sys.argv[1]
rho = float(sys.argv[2])
auc_threshold = float(sys.argv[3])

# Load and preprocess the dataset
df = select_data(dataset_path)

# Plot final accuracy vs. window size
plot_final_accuracy_vs_window_size(df, rho, auc_threshold)
