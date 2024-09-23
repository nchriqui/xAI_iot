import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.eddm import EDDM
import time
import sys

def select_data(x):
    df = pd.read_csv(x)
    scaler = MinMaxScaler()
    df.iloc[:, 0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:, 0:df.shape[1]-1])
    return df

def check_true(y, y_hat):
    return 1 if y == y_hat else 0

# Main function logic
df = select_data(sys.argv[1])
stream = DataStream(df)
stream_clf_EDDM = HoeffdingTreeClassifier()

w = int(sys.argv[2])
rho = float(sys.argv[3])

# Initialize EDDM
eddm = EDDM()

stream_acc_EDDM = []
stream_record_EDDM = []
stream_true_EDDM = 0

i = 0
start = time.time()
X, y = stream.next_sample(int(w * rho))

stream_clf_EDDM.partial_fit(X, y, classes=stream.target_values)

while stream.has_more_samples():
    X, y = stream.next_sample()    

    # EDDM Detector
    y_hat_EDDM = stream_clf_EDDM.predict(X)
    eddm.add_element(int(y_hat_EDDM == y))
    if eddm.detected_change():
        stream_clf_EDDM.reset()

    stream_true_EDDM += check_true(y, y_hat_EDDM)
    stream_clf_EDDM.partial_fit(X, y)
    stream_acc_EDDM.append(stream_true_EDDM / (i + 1))
    stream_record_EDDM.append(check_true(y, y_hat_EDDM))
    
    i += 1

elapsed = format(time.time() - start, '.4f')

acc_EDDM = format((stream_acc_EDDM[-1] * 100), '.4f')

print(f"Final accuracy - EDDM: {acc_EDDM}, Elapsed time: {elapsed}")

def window_average(x, N):
    low_index = 0
    high_index = low_index + N
    w_avg = []
    while high_index < len(x):
        temp = sum(x[low_index:high_index]) / N
        w_avg.append(temp)
        low_index += N
        high_index += N
    return w_avg

a = int(len(df) / 30)

ddd_acc_EDDM = window_average(stream_record_EDDM, a)

x = np.linspace(0, 100, len(ddd_acc_EDDM), endpoint=True)

f = plt.figure()

plt.plot(x, ddd_acc_EDDM, 'b', label='EDDM', marker="s")

plt.xlabel('Percentage of data', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True)
plt.legend(loc='lower left')

plt.show()
