import pandas as pd

# Load the dataset using pandas
data = pd.read_csv('data/all_annotated.tsv',sep='\t')

# Display basic info
print(data.info())
print(data.head())

# Save cleaned data (optional)
data.to_csv('E:\Projects\Lang\data\cleaned_data.csv', sep='\t', index= False)
