import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
df = pd.read_csv('concept-drift-datasets-scikit-multiflow-master/artificial/example.csv')

# Iterate over each column in the dataframe
for column in df.columns:
    plt.figure(figsize=(8, 6))  # Create a new figure for each plot
    plt.plot(df[column])
    
    # Add title and labels
    plt.title(f'Plot of {column}')
    plt.xlabel('Data')
    plt.ylabel(column)
    
    # Show the plot
    plt.show()


