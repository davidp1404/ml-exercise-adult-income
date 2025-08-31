import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import yaml

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['train']

# Load processed data
data = pd.read_csv('data/processed/train.csv')
X = data.drop(columns=['income'])
y = data['income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['test_size'], random_state=params['random_state']
)

# Train model
model = RandomForestClassifier(
    n_estimators=params['n_estimators'],
    max_depth=params['max_depth'],
    min_samples_split=params['min_samples_split'],
    min_samples_leaf=params['min_samples_leaf'],
    random_state=params['random_state']
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save training metrics
import json
metrics = {"accuracy": accuracy}
with open('metrics/train_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save model
joblib.dump(model, 'models/model.pkl')