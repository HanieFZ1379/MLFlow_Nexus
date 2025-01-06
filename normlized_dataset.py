import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample data: list of features for each sample
data = [
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
    [69, 0, 3, 140, 239, 0, 1, 151, 0, 1.8, 2, 2, 2],
    [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2],
    [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2],
    [71, 0, 1, 160, 302, 0, 1, 162, 0, 0.4, 2, 2, 2],
    [35, 1, 0, 120, 198, 0, 1, 130, 1, 1.6, 1, 0, 3],
    [52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3],
    [67, 0, 0, 106, 223, 0, 1, 142, 0, 0.3, 2, 2, 2],
    [60, 1, 2, 140, 185, 0, 0, 155, 0, 3, 1, 0, 2],
    [42, 0, 0, 102, 265, 0, 0, 122, 0, 0.6, 1, 0, 2],
    [51, 0, 0, 130, 305, 0, 1, 142, 1, 1.2, 1, 0, 3],
    [39, 1, 0, 118, 219, 0, 1, 140, 0, 1.2, 1, 0, 3],
    [68, 1, 0, 144, 193, 1, 1, 141, 0, 3.4, 1, 2, 3]
]

# Define column names
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Normalize the data (exclude non-numeric columns if necessary)
scaler = MinMaxScaler()

# Columns to normalize (numerical features only)
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Apply scaling
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save the normalized dataset to CSV
df.to_csv('normlized_dataset.csv', index=False)

print("Normalized dataset saved as 'normlized_dataset.csv'")
