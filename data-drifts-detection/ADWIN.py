import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
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
stream_clf_ADWIN = HoeffdingTreeClassifier()

w = int(sys.argv[2])
rho = float(sys.argv[3])

# Initialize ADWIN
adwin = ADWIN()

stream_acc_ADWIN = []
stream_record_ADWIN = []
stream_true_ADWIN = 0

i = 0
start = time.time()
X, y = stream.next_sample(int(w * rho))
stream_clf_ADWIN.partial_fit(X, y, classes=stream.target_values)

while stream.has_more_samples():
    X, y = stream.next_sample()    

    # ADWIN Detector
    y_hat_ADWIN = stream_clf_ADWIN.predict(X)
    adwin.add_element(int(y_hat_ADWIN == y))
    if adwin.detected_change():
        stream_clf_ADWIN.reset()

        # X_retrain, y_retrain = stream.next_sample(int(w * rho))
        # stream_clf_ADWIN.partial_fit(X_retrain, y_retrain, classes=stream.target_values)
    stream_true_ADWIN += check_true(y, y_hat_ADWIN)
    stream_clf_ADWIN.partial_fit(X, y)
    stream_acc_ADWIN.append(stream_true_ADWIN / (i + 1))
    stream_record_ADWIN.append(check_true(y, y_hat_ADWIN))

    i += 1

elapsed = format(time.time() - start, '.4f')
acc_ADWIN = format((stream_acc_ADWIN[-1] * 100), '.4f')

print(f"Final accuracy - ADWIN: {acc_ADWIN}, Elapsed time: {elapsed}")

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
ddd_acc_ADWIN = window_average(stream_record_ADWIN, a)


x = np.linspace(0, 100, len(ddd_acc_ADWIN), endpoint=True)

f = plt.figure()
plt.plot(x, ddd_acc_ADWIN, 'g', label='ADWIN', marker="o")

plt.xlabel('Percentage of data', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True)
plt.legend(loc='lower left')

plt.show()
