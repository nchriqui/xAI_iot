import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.eddm import EDDM
import time
import sys
from scipy.stats import pearsonr

# Calculate Kullback-Leibler Divergence
def kl_divergence(P, Q, epsilon=1e-10):
    P = np.clip(P, epsilon, 1)  # Avoid division by zero
    Q = np.clip(Q, epsilon, 1)
    return np.sum(P * np.log(P / Q))

# Calculate Hellinger Distance
def hellinger_distance(P, Q):
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2))

# Hellinger Distance and KL-divergence Calculation
def metrics_calc(S, T):
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    T['in_target'] = 0  # in target set
    S['in_target'] = 1  # in source set

    # Calculate Hellinger Distance
    S_distribution = np.mean(S, axis=0)
    T_distribution = np.mean(T, axis=0)
    h_distance = hellinger_distance(S_distribution, T_distribution)
    kl_div = kl_divergence(S_distribution, T_distribution)

    return h_distance, kl_div

# Class metrics
class HKL():
    def __init__(self, w, rho, dim):
        self.size = int(w*(1+rho))
        self.win_data = np.zeros((self.size,dim))
        self.win_label = np.zeros(self.size)
        self.w = w
        self.rho = rho
        self.dim = dim
        self.window_index = 0
        self.hellinger_distances = []  # To store Hellinger Distance values
        self.kl_divs = []  # To store KL Divergence values
        
    def addInstance(self,X,y):
        if self.isEmpty():
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.window_index += 1
        else:
            print("Error: Buffer is full!")
    
    def isEmpty(self):
        return self.window_index < self.size
    
    def update(self):
        h_distance, kl_div = metrics_calc(self.win_data[:self.w], self.win_data[self.w:self.size])
        self.hellinger_distances.append(h_distance)
        self.kl_divs.append(kl_div)
        
        self.window_index = self.w
        self.win_data = np.roll(self.win_data, -int(self.w * self.rho), axis=0)
        self.win_label = np.roll(self.win_label, -int(self.w * self.rho), axis=0)
    
    def getCurrentData(self):
        return self.win_data[:self.window_index]
    
    def getCurrentLabels(self):
        return self.win_label[:self.window_index]

def select_data(x):
    df = pd.read_csv(x)
    scaler = MinMaxScaler()
    df.iloc[:, 0:df.shape[1]-1] = scaler.fit_transform(df.iloc[:, 0:df.shape[1]-1])
    return df

def check_true(y, y_hat):
    return 1 if y == y_hat else 0

# Initialize parameters
df = select_data(sys.argv[1])
stream = DataStream(df)
stream_clf = HoeffdingTreeClassifier()
w = int(sys.argv[2])
rho = float(sys.argv[3])

HKL_win = HKL(w, rho, stream.n_features)

i = 0
X, y = stream.next_sample(int(w * rho))
stream_clf.partial_fit(X, y, classes=stream.target_values)
while stream.has_more_samples():
    X, y = stream.next_sample()
    if HKL_win.isEmpty():
        HKL_win.addInstance(X, y)
        stream_clf.partial_fit(X, y)
    else:
        HKL_win.update()
        stream_clf.reset()
        stream_clf.partial_fit(X, y)
        HKL_win.addInstance(X, y)
    i += 1  



# Main function logic
stream = DataStream(df)
stream_clf_EDDM = HoeffdingTreeClassifier()

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

x = np.arange(a, a * len(ddd_acc_EDDM) + 1, a)

# Calculate correlation between accuracy and Hellinger distance and accuracy and KL-divergence
# Ensure that both lists have the same length by trimming the lists
trimmed_hellinger_distances = HKL_win.hellinger_distances[:len(x)]
trimmed_kl_divs = HKL_win.kl_divs[:len(x)]

correlation, p_value = pearsonr(ddd_acc_EDDM, trimmed_hellinger_distances)
print(f"Correlation between Accuracy and Hellinger Distance: {correlation}, p-value: {p_value}")

correlation, p_value = pearsonr(ddd_acc_EDDM, trimmed_kl_divs)
print(f"Correlation between Accuracy and KL Divergence: {correlation}, p-value: {p_value}")


# Plot Accuracy and Hellinger Distance
f, ax1 = plt.subplots()

ax1.plot(x, ddd_acc_EDDM, 'r', label='ADWIN Accuracy', marker="*")
ax1.set_xlabel('Processed Data Points')
ax1.set_ylabel('Accuracy', color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()  
ax2.plot(x, HKL_win.hellinger_distances[:len(x)], 'b', label='Hellinger Distance', marker="o")
ax2.set_ylabel('Hellinger Distance', color='b')
ax2.tick_params(axis='y', labelcolor='b')

f.tight_layout()  
plt.grid(True)
plt.show()

# Plot Accuracy and KL-divergence
f2, ax1 = plt.subplots()

ax1.plot(x, ddd_acc_EDDM, 'r', label='D3 Accuracy', marker="*")
ax1.set_xlabel('Processed Data Points')
ax1.set_ylabel('Accuracy', color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()
ax2.plot(x, HKL_win.kl_divs[:len(x)], 'g', label='KL Divergence', marker="^")
ax2.set_ylabel('KL Divergence', color='g')
ax2.tick_params(axis='y', labelcolor='g')

f2.tight_layout()  
plt.grid(True)
plt.show()
