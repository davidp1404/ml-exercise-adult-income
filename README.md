# Adult Income Prediction

A machine learning project that predicts whether an individual's income exceeds $50K annually using the UCI Adult Census dataset.

## What This Exercise Covers

This project demonstrates key MLOps practices and tools:

### Data Version Control (DVC)
- **Pipeline Management**: Automated ML pipeline with dependency tracking
- **Reproducible Experiments**: Parameterized training with `params.yaml`
- **Metrics Tracking**: Automated collection and versioning of model performance
- **Data Versioning**: Track changes to datasets and model artifacts

### Machine Learning Pipeline
- **Data Preprocessing**: Feature scaling, encoding, and train/test splitting
- **Model Training**: RandomForest classifier with configurable hyperparameters
- **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Feature Analysis**: Permutation importance for interpretability

### Pipeline Stages
1. **preprocess**: Clean and prepare raw census data
2. **train**: Train RandomForest model with tracked parameters
3. **evaluate**: Generate performance metrics on test set
4. **permutation_importance**: Analyze feature importance

### Visualization & Monitoring
- **DVC Plots**: Interactive charts for metrics over time
- **Feature Importance**: Horizontal bar charts showing most predictive features
- **Experiment Tracking**: Compare performance across parameter changes

### Key Technologies
- **scikit-learn**: ML algorithms and preprocessing
- **DVC**: Pipeline orchestration and experiment tracking
- **pandas**: Data manipulation
- **Python**: Core implementation

## Usage

### Quick Start
```bash
# Create venv with dependencies
uv sync

# Populate data/raw folder
uv run src/download_dataset.py

# Run pipeline
uv run dvc repro
```

### Restoring from local dvc-remote
This approach doesn't make sense because includes the dvc data in git, that is what
DVC resolve pushing the data to an external repo (s3, azure blob, etc..).
This only helps to put in practice the process assuming an external remote

```bash
# Remove previous reference having complete path
dvc remote remove localremote

# Recreate the remote to where dvc stored the data
dvc remote add -d localremote $(pwd)/dvc-storage

# Restore data from dvc repo
dvc pull
```

### Running Experiments
```bash
# Run experiment with custom hyperparameters
dvc exp run n_estimators=200 max_depth=10

# Run experiment with different parameters
dvc exp run n_estimators=50 min_samples_split=5

# Compare with previous experiment
dvc exp show
```

### DVC Experiments (using existing Makefile)
```bash
# Queue multiple experiments with different parameters
make experiments

# Run queued experiments in parallel
make run

# Clean experiment queue
make clean
```

### Direct DVC Commands
```bash
# Activate venv
source .venv/bin/activate

# Run full pipeline
dvc repro

# View metrics
dvc metrics show

# Generate plots
dvc plots show

# Modify hyperparameters in params.yaml and rerun
dvc repro
```

## Results

The model achieves ~85% accuracy with capital-gain being the most important feature for income prediction.
