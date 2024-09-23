# Internship Research: Enabling XAI in IoT-enhanced Spaces

The objective of this internship is to explain data drifts (virtual drifts): changes in the input data distribution:

1. Detect data drift: measure difference between input distributions using metrics: Hellinger Distance and KL-Divergence. 
2. Correlate data drift with concept drift: understand when and how changes in input data distributions lead to changes in the predictive model
3. Explain the impact of data drift: identify features responsible for the shift and how they contribute to model performance


## First Approach: Using HDDDM

The first approach use HDDDM detector located in HDDDM folder ([here](./HDDDM/)). Tests are realized with synthetical data.

## Second Approach

The second approach use Concept Drifts Detectors and detects Data Drifts to explain Concept Drifts Detectors located in data-drifts-detection folder ([here](./data-drifts-detection/)).

### Datasets

The dataset used are in the concept-drift-datasets-scikit-multiflow-master folder ([here](./concept-drift-datasets-scikit-multiflow-master/)).

### Run

Detectors D3, ADWIN and EDDM are used with their original results and with the Hellinger distance and KL-divergence.
To run D3 use the command: 
```
python .\data-drifts-detection\<detector.py> <dataset> <size of the old data> <percentage of new data with respect to old> <threshold for auc>
```
Example:
```
python .\data-drifts-detection\D3.py .\concept-drift-datasets-scikit-multiflow-master\real-world\elec.csv 100 0.1 0.7
```

To run ADWIN or EDDM use the command: 
```
python .\data-drifts-detection\<detector.py> <dataset> <size of the old data> <percentage of new data with respect to old>
```
Example:
```
python .\data-drifts-detection\ADWIN.py .\concept-drift-datasets-scikit-multiflow-master\real-world\elec.csv 100 0.1
```

And too see Hellinger distance and KL-divergence the commands are the same but with "_hellinger_kl" for each detector name.
Example:
```
python .\data-drifts-detection\D3_hellinger_kl.py .\concept-drift-datasets-scikit-multiflow-master\real-world\elec.csv 100 0.1 0.7
```

The code to generate the synthetic dataset is also located in the folder and you can also plot datasets to see the drifts.

### Results

Finally the results found during this internship are located in the folder [results](./data-drifts-detection/results/)