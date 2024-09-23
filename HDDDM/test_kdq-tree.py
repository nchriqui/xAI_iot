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

# ## KDQ-Tree Detection Method (Streaming Setting)

# KdqTree monitors incoming features by constructing a tree which partitions the feature-space, and then monitoring a divergence statistic that is defined over that partition. It watches data within a sliding window of a particular size. When that window is full, it builds the reference tree. As the window moves forward, point-by-point, the data in that new window is compared against the reference tree to detect drift.

# In[ ]:


## Setup ##

# kdqTree does use bootstrapping to define its critical thresholds, so setting
# the seed is important to reproduce exact behavior.
np.random.seed(1)

# Note that the default input_type for KDQTree is "stream".
# The window size, corresponding to the portion of the stream which KDQTree
# monitors, must be specified.
det = KdqTreeStreaming(window_size=500, alpha=0.05, bootstrap_samples=500, count_ubound=50)

# setup DF to record results
status = pd.DataFrame(columns=["index", "var1", "var2", "drift_detected"])

# iterate through X data and run detector
data = circle_data[["var1", "var2"]]


# In[ ]:


## Plotting ##

plot_data = {}
for i in range(len(circle_data)):
    det.update(data.iloc[[i]])
    status.loc[i] = [i, data.iloc[i, 0], data.iloc[i, 1], det.drift_state]
    if det.drift_state is not None:
        # capture the visualization data
        plot_data[i] = det.to_plotly_dataframe()

plt.figure(figsize=(20, 6))
plt.scatter("index", "var2", data=status, label="var2")
plt.scatter("index", "var1", data=status, label="var1", alpha=0.5)
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("KDQ Tree Test Results", fontsize=22)
plt.ylabel("Value", fontsize=18)
plt.xlabel("Index", fontsize=18)
ylims = [-0.05, 1.05]
plt.ylim(ylims)

drift_start, drift_end = 1000, 1250
plt.axvspan(drift_start, drift_end, alpha=0.5, label="Drift Induction Window")

# Draw red lines that indicate where drift was detected
plt.vlines(
    x=status.loc[status["drift_detected"] == "drift"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)

plt.legend()
plt.show()
# plt.savefig("example_streaming_kdqtree_feature_stream.png")


# Given a window_size of 500, with only the two input features, KdqTree detects
# a change after about half of the data within its window is in the new regime.
# 

# If we save off the to_plotly_dataframe at each drift detection, we can display
# the Kulldorff Spatial Scan Statistic (KSS) for each. Higher values of KSS
# indicate that a given region of the data space has greater divergence between
# the reference and test data.
# 
# Note that the structure of the particular tree depends on the reference data
# and the order of the columns within the dataframe!
# 
# Since this data only contains two features, the tree is relatively
# shallow.

# In[ ]:


# ## Kulldorff Spatial Scan Statistic (KSS) ##
# for title, df_plot in plot_data.items():
#     fig = px.treemap(
#         data_frame=df_plot,
#         names="name",
#         ids="idx",
#         parents="parent_idx",
#         color="kss",
#         color_continuous_scale="blues",
#         title=f"Index {title}",
#     )
#     fig.update_traces(root_color="lightgrey")
#     fig.show()
#     # fig.write_html(f"example_streaming_kdqtree_treemap_{title}.html")