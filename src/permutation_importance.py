import pandas as pd
import joblib
import json
from sklearn.inspection import permutation_importance

# Load test data and model
data = pd.read_csv('data/processed/test.csv')
model = joblib.load('models/model.pkl')

X = data.drop(columns=['income'])
y = data['income']

# Calculate permutation importance
perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

# Create results
feature_importance = {
    'features': X.columns.tolist(),
    'importance_mean': perm_importance.importances_mean.tolist(),
    'importance_std': perm_importance.importances_std.tolist()
}

# Save results
with open('metrics/permutation_importance.json', 'w') as f:
    json.dump(feature_importance, f, indent=2)

# Print top features
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
print(importance_df.head(10).to_string(index=False))
