import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
import time
import sys

# Drift Detector
# S: Source (Old Data)
# T: Target (New Data)
# ST: S&T combined
def drift_detector(S,T,threshold = 0.75):
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    # Give slack variable in_target which is 1 for old and 0 for new
    T['in_target'] = 0 # in target set
    S['in_target'] = 1 # in source set
    # Combine source and target with new slack variable 
    ST = pd.concat( [T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values
    # You can use any classifier for this step. We advise it to be a simple one as we want to see whether source
    # and target differ not to classify them.
    clf = LogisticRegression(solver='liblinear')
    predictions = np.zeros(labels.shape)
    # Divide ST into two equal chunks
    # Train LR on a chunk and classify the other chunk
    # Calculate AUC for original labels (in_target) and predicted ones
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = AUC(labels, predictions)
    # Signal drift if AUC is larger than the threshold
    if auc_score > threshold:
        return True
    else:
        return False


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
    def addInstance(self,X,y):
        if(self.isEmpty()):
            self.win_data[self.window_index] = X
            self.win_label[self.window_index] = y
            self.window_index = self.window_index + 1
        else:
            print("Error: Buffer is full!")
    def isEmpty(self):
        return self.window_index < self.size
    def driftCheck(self):
        if drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size], auc): #returns true if drift is detected
            self.window_index = int(self.w * self.rho)
            self.win_data = np.roll(self.win_data, -1*self.w, axis=0)
            self.win_label = np.roll(self.win_label, -1*self.w, axis=0)
            self.drift_count = self.drift_count + 1
            return True
        else:
            self.window_index = self.w
            self.win_data = np.roll(self.win_data, -1*(int(self.w*self.rho)), axis=0)
            self.win_label =np.roll(self.win_label, -1*(int(self.w*self.rho)), axis=0)
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

# Main function logic
df = select_data(sys.argv[1])
stream = DataStream(df)
stream_clf_D3 = HoeffdingTreeClassifier()
stream_clf_ADWIN = HoeffdingTreeClassifier()
stream_clf_EDDM = HoeffdingTreeClassifier()
stream_clf_DDM = HoeffdingTreeClassifier()

w = int(sys.argv[2])
rho = float(sys.argv[3])
auc = float(sys.argv[4])

# Initialize D3 window and other detectors
D3_win = D3(w, rho, stream.n_features, auc)
adwin = ADWIN()
eddm = EDDM()
ddm = DDM()

stream_acc_D3 = []
stream_acc_ADWIN = []
stream_acc_EDDM = []
stream_acc_DDM = []

stream_record_D3 = []
stream_record_ADWIN = []
stream_record_EDDM = []
stream_record_DDM = []

stream_true_D3 = stream_true_ADWIN = stream_true_EDDM = stream_true_DDM = 0

i = 0
start = time.time()
X, y = stream.next_sample(int(w * rho))
stream_clf_D3.partial_fit(X, y, classes=stream.target_values)
stream_clf_ADWIN.partial_fit(X, y, classes=stream.target_values)
stream_clf_EDDM.partial_fit(X, y, classes=stream.target_values)
stream_clf_DDM.partial_fit(X, y, classes=stream.target_values)

while stream.has_more_samples():
    X, y = stream.next_sample()

    # D3 Detector
    if D3_win.isEmpty():
        D3_win.addInstance(X, y)
    else:
        if D3_win.driftCheck():
            stream_clf_D3.reset()
            stream_clf_D3.partial_fit(D3_win.getCurrentData(), D3_win.getCurrentLabels(), classes=stream.target_values)
        D3_win.addInstance(X, y)
    
    y_hat_D3 = stream_clf_D3.predict(X)
    stream_true_D3 += check_true(y, y_hat_D3)
    stream_clf_D3.partial_fit(X, y)
    stream_acc_D3.append(stream_true_D3 / (i + 1))
    stream_record_D3.append(check_true(y, y_hat_D3))

    # ADWIN Detector
    y_hat_ADWIN = stream_clf_ADWIN.predict(X)
    adwin.add_element(int(y_hat_ADWIN == y))
    if adwin.detected_change():
        stream_clf_ADWIN.reset()
    stream_true_ADWIN += check_true(y, y_hat_ADWIN)
    stream_clf_ADWIN.partial_fit(X, y)
    stream_acc_ADWIN.append(stream_true_ADWIN / (i + 1))
    stream_record_ADWIN.append(check_true(y, y_hat_ADWIN))

    # EDDM Detector
    y_hat_EDDM = stream_clf_EDDM.predict(X)
    eddm.add_element(int(y_hat_EDDM == y))
    if eddm.detected_change():
        stream_clf_EDDM.reset()
    stream_true_EDDM += check_true(y, y_hat_EDDM)
    stream_clf_EDDM.partial_fit(X, y)
    stream_acc_EDDM.append(stream_true_EDDM / (i + 1))
    stream_record_EDDM.append(check_true(y, y_hat_EDDM))

    # DDM Detector
    y_hat_DDM = stream_clf_DDM.predict(X)
    ddm.add_element(int(y_hat_DDM == y))
    if ddm.detected_change():
        stream_clf_DDM.reset()
    stream_true_DDM += check_true(y, y_hat_DDM)
    stream_clf_DDM.partial_fit(X, y)
    stream_acc_DDM.append(stream_true_DDM / (i + 1))
    stream_record_DDM.append(check_true(y, y_hat_DDM))

    i += 1

elapsed = format(time.time() - start, '.4f')
acc_D3 = format((stream_acc_D3[-1] * 100), '.4f')
acc_ADWIN = format((stream_acc_ADWIN[-1] * 100), '.4f')
acc_EDDM = format((stream_acc_EDDM[-1] * 100), '.4f')
acc_DDM = format((stream_acc_DDM[-1] * 100), '.4f')

print(f"Final accuracy - D3: {acc_D3}, ADWIN: {acc_ADWIN}, EDDM: {acc_EDDM}, DDM: {acc_DDM}, Elapsed time: {elapsed}")

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
ddd_acc_D3 = window_average(stream_record_D3, a)
ddd_acc_ADWIN = window_average(stream_record_ADWIN, a)
ddd_acc_EDDM = window_average(stream_record_EDDM, a)
ddd_acc_DDM = window_average(stream_record_DDM, a)

x = np.linspace(0, 100, len(ddd_acc_D3), endpoint=True)

f = plt.figure()
plt.plot(x, ddd_acc_D3, 'r', label='D3', marker="*")
plt.plot(x, ddd_acc_ADWIN, 'g', label='ADWIN', marker="o")
plt.plot(x, ddd_acc_EDDM, 'b', label='EDDM', marker="s")
plt.plot(x, ddd_acc_DDM, 'y', label='DDM', marker="^")

plt.xlabel('Percentage of data', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True)
plt.legend(loc='lower left')

plt.show()
# f.savefig("drift_detection_comparison.pdf", bbox_inches='tight')
