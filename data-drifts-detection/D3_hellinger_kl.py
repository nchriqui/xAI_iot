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
from scipy.stats import pearsonr

# Calculate Kullback-Leibler Divergence
def kl_divergence(P, Q, epsilon=1e-10):
    P = np.clip(P, epsilon, 1)  # Avoid division by zero
    Q = np.clip(Q, epsilon, 1)
    return np.sum(P * np.log(P / Q))

# Calculate Hellinger Distance
def hellinger_distance(P, Q):
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2))

# Calculate Hellinger Distance for each feature separately using summary statistics (mean)
def hellinger_distance_per_feature(S, T):
    distances = []
    
    for i in range(S.shape[1]):
        # Summarize the feature by computing the mean for each feature in both windows
        mean_S = np.mean(S[:, i])
        mean_T = np.mean(T[:, i])
        
        # Compute the Hellinger distance using the means
        distances.append(np.sqrt(0.5 * (np.sqrt(mean_S) - np.sqrt(mean_T)) ** 2))
    
    return distances

# Drift Detector with Hellinger Distance and KL-divergence Calculation
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
    coef_importance = np.zeros(ST.shape[1])
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
        coef_importance += np.abs(clf.coef_[0])
    
    auc_score = AUC(labels, predictions)

    coef_importance /= skf.get_n_splits()

    # Calculate Hellinger Distance and KL-divergence
    S_distribution = np.mean(S, axis=0)
    T_distribution = np.mean(T, axis=0)
    h_distance = hellinger_distance(S_distribution, T_distribution)
    kl_div = kl_divergence(S_distribution, T_distribution)

    h_distances_per_feature = hellinger_distance_per_feature(S.values, T.values)

    if auc_score > threshold:
        # print(f"AUC Score: {auc_score}, Hellinger Distance: {h_distance}")
        return True, h_distance, kl_div, coef_importance, h_distances_per_feature
    else:
        return False, h_distance, kl_div, coef_importance, h_distances_per_feature

class D3():
    def __init__(self, w, rho, dim, auc):
        self.size = int(w*(1+rho))
        self.win_data = np.zeros((self.size,dim))
        self.win_label = np.zeros(self.size)
        self.w = w
        self.rho = rho
        self.dim = dim
        self.auc = auc
        self.drift_count = 0
        self.window_index = 0
        self.hellinger_distances = []  # To store Hellinger Distance values
        self.kl_divs = []  # To store KL Divergence values
        self.feature_importances = []  # To store feature importance values
        self.hellinger_distances_per_feature = [] # To store hellinger distance per feature values
        
    def addInstance(self,X,y):
        if self.isEmpty():
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.window_index += 1
        else:
            print("Error: Buffer is full!")
    
    def isEmpty(self):
        return self.window_index < self.size
    
    def driftCheck(self):
        drift_detected, h_distance, kl_div, coef_importance, h_distances_per_feature = drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size], self.auc)
        self.hellinger_distances.append(h_distance)
        self.kl_divs.append(kl_div)
        self.feature_importances.append(coef_importance)
        self.hellinger_distances_per_feature.append(h_distances_per_feature)
        if drift_detected:
            self.window_index = int(self.w * self.rho)
            self.win_data = np.roll(self.win_data, -self.w, axis=0)
            self.win_label = np.roll(self.win_label, -self.w, axis=0)
            self.drift_count += 1
            return True
        else:
            self.window_index = self.w
            self.win_data = np.roll(self.win_data, -int(self.w * self.rho), axis=0)
            self.win_label = np.roll(self.win_label, -int(self.w * self.rho), axis=0)
            return False
    
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

# python .\data-drifts-detection\D3_hellinger_kl.py <dataset> <size of the old data> <percentage of new data with respect to old> <threshold for auc>

# Example:
# python .\data-drifts-detection\D3_hellinger_kl.py .\concept-drift-datasets-scikit-multiflow-master\real-world\elec.csv 100 0.1 0.7
df = select_data(sys.argv[1])
stream = DataStream(df)
stream_clf = HoeffdingTreeClassifier()
w = int(sys.argv[2])
rho = float(sys.argv[3])
auc = float(sys.argv[4])

D3_win = D3(w, rho, stream.n_features, auc)
stream_acc = []
stream_record = []
stream_true = 0

i = 0
start = time.time()
X, y = stream.next_sample(int(w * rho))
stream_clf.partial_fit(X, y, classes=stream.target_values)
while stream.has_more_samples():
    X, y = stream.next_sample()
    if D3_win.isEmpty():
        D3_win.addInstance(X, y)
        y_hat = stream_clf.predict(X)
        stream_true += check_true(y, y_hat)
        stream_clf.partial_fit(X, y)
        stream_acc.append(stream_true / (i + 1))
        stream_record.append(check_true(y, y_hat))
    else:
        if D3_win.driftCheck():  # Detected drift
            stream_clf.reset()
            stream_clf.partial_fit(D3_win.getCurrentData(), D3_win.getCurrentLabels(), classes=stream.target_values)
            y_hat = stream_clf.predict(X)
            stream_true += check_true(y, y_hat)
            stream_clf.partial_fit(X, y)
            stream_acc.append(stream_true / (i + 1))
            stream_record.append(check_true(y, y_hat))
            D3_win.addInstance(X, y)
        else:
            y_hat = stream_clf.predict(X)
            stream_true += check_true(y, y_hat)
            stream_clf.partial_fit(X, y)
            stream_acc.append(stream_true / (i + 1))
            stream_record.append(check_true(y, y_hat))
            D3_win.addInstance(X, y)
    i += 1  

elapsed = format(time.time() - start, '.4f')
acc = format((stream_acc[-1] * 100), '.4f')
final_accuracy = f"Final accuracy: {acc}, Elapsed time: {elapsed}"
print(final_accuracy)


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
ddd_acc2 = window_average(stream_record, a)

x = np.arange(a, a * len(ddd_acc2) + 1, a)

# Calculate correlation between accuracy and Hellinger distance and accuracy and KL-divergence
# Ensure that both lists have the same length by trimming the lists
trimmed_hellinger_distances = D3_win.hellinger_distances[:len(x)]
trimmed_kl_divs = D3_win.kl_divs[:len(x)]

correlation, p_value = pearsonr(ddd_acc2, trimmed_hellinger_distances)
print(f"Correlation between Accuracy and Hellinger Distance: {correlation}, p-value: {p_value}")

correlation, p_value = pearsonr(ddd_acc2, trimmed_kl_divs)
print(f"Correlation between Accuracy and KL Divergence: {correlation}, p-value: {p_value}")


# Plot Accuracy and Hellinger Distance
f, ax1 = plt.subplots()

ax1.plot(x, ddd_acc2, 'r', label='D3 Accuracy', marker="*")
ax1.set_xlabel('Processed Data Points')
ax1.set_ylabel('Accuracy', color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()
ax2.plot(x, D3_win.hellinger_distances[:len(x)], 'b', label='Hellinger Distance', marker="o")
ax2.set_ylabel('Hellinger Distance', color='b')
ax2.tick_params(axis='y', labelcolor='b')

f.tight_layout()  
plt.grid(True)
plt.show()

# Plot Accuracy and KL-divergence
f2, ax1 = plt.subplots()

ax1.plot(x, ddd_acc2, 'r', label='D3 Accuracy', marker="*")
ax1.set_xlabel('Processed Data Points')
ax1.set_ylabel('Accuracy', color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()
ax2.plot(x, D3_win.kl_divs[:len(x)], 'g', label='KL Divergence', marker="^")
ax2.set_ylabel('KL Divergence', color='g')
ax2.tick_params(axis='y', labelcolor='g')

f2.tight_layout()  
plt.grid(True)
plt.show()

# Plot feature importance after drift detection
def plot_feature_importance(D3_win, feature_names):
    drift_points = np.arange(len(D3_win.feature_importances))
    
    for i in range(len(D3_win.feature_importances[0])):
    # for i in range(len(D3_win.feature_importances[0])-1): # For example data (synthetic data)

        # Feature importance per feature per drift
        feature_importance_per_drift = [imp[i] for imp in D3_win.feature_importances]
        plt.plot(drift_points, feature_importance_per_drift, label=f'Feature {feature_names[i]}')
    
    plt.xlabel('Drift Points')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance during Drift Detection')
    plt.legend(loc='best')
    plt.show()

feature_names = df.columns[:-1]  # Assuming last column is the label
# feature_names = df.columns[:-2] # For example data (synthetic data): confidence and drfit col
plot_feature_importance(D3_win, feature_names)

# Plot Hellinger distance per feature after drift detection
def plot_feature_hellinger(D3_win, feature_names):
    drift_points = np.arange(len(D3_win.feature_importances))
    
    for i in range(len(D3_win.feature_importances[0])):
    # for i in range(len(D3_win.feature_importances[0])-1): # For example data (synthetic data)

        # Hellinger distance per feature per drift
        hellinger_per_drift = [h_dist[i] for h_dist in D3_win.hellinger_distances_per_feature]
        plt.plot(drift_points, hellinger_per_drift, label=f'Hellinger Distance: {feature_names[i]}', marker="o")
        
        
    plt.xlabel('Drift Points')
    plt.ylabel('Hellinger Distance')
    plt.title(f'Hellinger Distance for Feature: {feature_names[i]}')
    plt.legend(loc='best')
    plt.show()

feature_names = df.columns[:-1]  # Assuming last column is the label
# feature_names = df.columns[:-2] # For example data (synthetic data): confidence and drfit col
plot_feature_hellinger(D3_win, feature_names)
