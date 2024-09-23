import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from menelaus.data_drift.cdbd import CDBD
from menelaus.data_drift.hdddm import HDDDM
from menelaus.data_drift import PCACD
from menelaus.data_drift import KdqTreeStreaming, KdqTreeBatch
from menelaus.data_drift import NNDVI
from menelaus.datasets import make_example_batch_data, fetch_circle_data

circle_data = fetch_circle_data()


# ## PCA-Based Change Detection (PCA-CD)

# PCA-CD is a drift detector that transforms the passed data into its principal components, then watches the transformed data for signs of drift by monitoring the KL-divergence via the Page-Hinkley algorithm.

# In[ ]:


## Setup ##

pca_cd = PCACD(window_size=50, divergence_metric="intersection")

# set up dataframe to record results
status = pd.DataFrame(columns=["index", "var1", "var2", "drift_detected"])

# Put together a dataframe of several features, each of which abruptly changes
# at index 1000.
np.random.seed(1)
size = 1000
data = pd.concat(
    [
        pd.DataFrame(
            [
                np.random.normal(1, 10, size),
                np.random.uniform(1, 2, size),
                np.random.normal(0, 1, size),
            ]
        ).T,
        pd.DataFrame(
            [
                np.random.normal(9, 10, size),
                np.random.normal(1, 3, size),
                np.random.gamma(20, 30, size),
            ]
        ).T,
    ]
)

# Update the drift detector with each new sample
for i in range(len(circle_data)):
    pca_cd.update(data.iloc[[i]])
    status.loc[i] = [i, data.iloc[i, 0], data.iloc[i, 1], pca_cd.drift_state]


# In[ ]:


## Plotting ##

# Plot the features and the drift
plt.figure(figsize=(20, 6))
plt.scatter(status.index, status.var2, label="Var 2")
plt.scatter(status.index, status.var1, label="Var 1", alpha=0.5)
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("PCA-CD Test Results", fontsize=22)
plt.ylabel("Value", fontsize=18)
plt.xlabel("Index", fontsize=18)
ylims = min(status.var1.min(), status.var2.min()), max(
    status.var1.max(), status.var1.max()
)
plt.ylim(ylims)

# Draw red lines that indicate where drift was detected
plt.vlines(
    x=status.loc[status["drift_detected"] == "drift"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)

plt.legend()


# PCA-CD detects this very abrupt drift within a few samples of its induction.
# 

# In[ ]:


plt.show()
# plt.savefig("example_PCA_CD.png")