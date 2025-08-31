import pandas as pd

# URLs for the dataset
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Column names
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Load training and test datasets
train_data = pd.read_csv(url_train, header=None, names=columns, na_values=" ?", skipinitialspace=True)
test_data = pd.read_csv(url_test, header=0, names=columns, na_values=" ?", skipinitialspace=True)

# Combine datasets
data = pd.concat([train_data, test_data], ignore_index=True)

# Save to CSV
data.to_csv('data/raw/adult.csv', index=False)