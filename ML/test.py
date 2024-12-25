import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.model_selection import train_test_split
from GaussianNaiveBayes import GaussianNaiveBayes
from KNearestNeighbors import KNearestNeighbors

def test_gaussian_naive_bayes():
    # Load iris dataset
    features, labels = load_iris(return_X_y=True)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.5, random_state=0
    )
    
    # Create and train model
    model = GaussianNaiveBayes()
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, predictions, average="macro")
    
    print("GaussianNaiveBayes Test Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {fscore}")
    print(f"Mislabeled points: {(predictions != test_labels).sum()}/{test_features.shape[0]}")

def test_knn():
    iris = load_iris()
    train_features, test_features, train_labels, test_labels = train_test_split(
        iris.data, iris.target, test_size=0.5, random_state=0
    )


    model = KNearestNeighbors(train_features, train_labels, k=3)
    predictions = model.predict(test_features)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, predictions, average="macro")

    print("KNearestNeighbors Test Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {fscore}")
    print(f"Mislabeled points: {(predictions != test_labels).sum()}/{test_features.shape[0]}")

    

def main():
    parser = argparse.ArgumentParser(description='Test different ML models')
    parser.add_argument('--name', type=str, required=True, 
                      choices=['gnb', 'knn'],
                      help='Name of the model to test (gnb: GaussianNaiveBayes, knn: KNearestNeighbor)')
    
    args = parser.parse_args()
    
    try:
        if args.name == 'gnb':
            test_gaussian_naive_bayes()
        elif args.name == 'knn':
            test_knn()
        else:
            print(f"Model {args.name} not implemented yet")
    except Exception as e:
        print(f"Error testing {args.name} model: {str(e)}")

if __name__ == "__main__":
    main()
