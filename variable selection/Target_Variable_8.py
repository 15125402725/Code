# pandas: Used for data reading and processing.
# sklearn.feature_selection.VarianceThreshold: Used for feature selection by applying the variance threshold method to select features with higher variance.

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Read the data
file_path = 'PRAD_LUAD.csv'
data = pd.read_csv(file_path)
# Extract the gene expression data part
gene_data = data.drop(columns=['y'])
# Use the variance threshold method for feature selection, setting the variance threshold
selector = VarianceThreshold(threshold=0.01)  # You can adjust the threshold as needed
selected_gene_data = selector.fit_transform(gene_data)
# Get the names of the selected features
selected_features = gene_data.columns[selector.get_support()]
# Rebuild the filtered data frame (including target variable 'y' and selected gene data)
filtered_data = pd.DataFrame(selected_gene_data, columns=selected_features)
filtered_data['y'] = data['y']
# Save the filtered data to a new CSV file
filtered_data.to_csv('PRAD_LUAD_filtered_gene_data.csv', index=False)
print("The filtered data has been saved as 'PRAD_LUAD_filtered_gene_data.csv'")
