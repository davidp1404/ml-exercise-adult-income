import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import json

# Load processed data and model
data = pd.read_csv('data/processed/test.csv')
model = joblib.load('models/model.pkl')

# Split data
X = data.drop(columns=['income'])
y = data['income']

# Predict and evaluate
y_pred = model.predict(X)
print(classification_report(y, y_pred))

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')

# Save evaluation metrics
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open('metrics/eval_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Create metrics history for plotting
import os
history_file = 'metrics/metrics_history.json'
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        history = json.load(f)
else:
    history = []

# Add current run
run_data = {
    "step": len(history),
    "train_accuracy": 0.84952,  # From train stage
    "eval_accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}
history.append(run_data)

with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)