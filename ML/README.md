# Machine Learning Models

This directory contains implementations of various machine learning models from scratch.

## Available Models

- **Gaussian Naive Bayes** (`GaussianNaiveBayes.py`)
  - A probabilistic classifier based on Bayes' theorem
  - Assumes features follow a normal distribution
  
- **K-Nearest Neighbors** (`KNearestNeighbors.py`)
  - Instance-based learning algorithm
  - Uses k closest training examples for classification

## Testing Models

Use `test.py` to evaluate model performance. The script supports different models and uses the iris dataset for testing.

### Usage

```bash
python test.py --name [model_name]
```

Available model names:
- `gnb`: Gaussian Naive Bayes
- `knn`: K-Nearest Neighbors

### Example Output

```
GaussianNaiveBayes Test Results:
Accuracy: 0.96
Precision: 0.96
Recall: 0.96
F1 Score: 0.96
Mislabeled points: 3/75
```

## Metrics

Each model evaluation includes:
- Accuracy
- Precision
- Recall
- F1 Score
- Number of mislabeled points

## Dataset

Currently using the iris dataset from scikit-learn for testing and evaluation.
