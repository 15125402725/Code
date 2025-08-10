import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Read the variance-filtered data
filtered_data = pd.read_csv('PRAD_LUAD_filtered_gene_data.csv')

# Extract the target variable and gene expression data
X = filtered_data.drop(columns=['y'])  # Gene expression data
y = filtered_data['y']  # Target variable

# Use the SIS method (SelectKBest) for feature selection, based on ANOVA F-value
selector_sis = SelectKBest(score_func=f_classif, k=100)  # Select the top 100 most relevant features
X_selected = selector_sis.fit_transform(X, y)

# Get the names of the selected features
selected_features_sis = X.columns[selector_sis.get_support()]

# Rebuild the filtered data frame (including target variable 'y' and selected gene data)
filtered_data_sis = pd.DataFrame(X_selected, columns=selected_features_sis)
filtered_data_sis['y'] = y

# Adjust 'y' to be the first column
cols = ['y'] + [col for col in filtered_data_sis.columns if col != 'y']
filtered_data_sis = filtered_data_sis[cols]

# Save the SIS filtered data to a new CSV file
filtered_data_sis.to_csv('PRAD_LUAD_filtered_gene_data_SIS.csv', index=False)
print("The second phase SIS filtered data has been saved as 'PRAD_LUAD_filtered_gene_data_SIS.csv', with 'y' as the first column.")
