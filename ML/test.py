import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from GaussianNaiveBayes import GaussianNaiveBayes
from KNearestNeighbors import KNearestNeighbors
from LinearRegression import LinearRegression


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

def test_lr():
    plt.style.use('bmh')
    train_features = np.arange(0, 250).reshape(-1, 1)
    train_labels = np.arange(0, 500, 2)

    test_features = np.arange(300, 400, 8).reshape(-1, 1)
    test_labels = np.arange(600, 800, 16)

    linear_regression = LinearRegression(epochs=25, learning_rate=1e-5, logging=False)
    linear_regression.fit(train_features, train_labels)
    predictions = linear_regression.predict(test_features).round()

    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("f(x) = 2x")
    fig.tight_layout()
    fig.set_size_inches(18, 8)

    axs[0].set_title("Visualization for f(x) = 2x")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].plot(train_features, train_labels)

    axs[1].set_title("Scatterplot for f(x) = 2x Data")
    axs[1].set_xlabel("x")
    axs[1].set_xlabel("y")
    axs[1].scatter(test_features, test_labels, color="blue")

    axs[2].set_title("Visualization for Approximated f(x) = 2x")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].scatter(test_features, test_labels, color="blue")
    axs[2].plot(test_features, predictions)

    plt.show()

    accuracy = accuracy_score(predictions, test_labels)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, predictions, average="macro")

    print("LinearRegression Test Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {fscore}")


    

def main():
    parser = argparse.ArgumentParser(description='Test different ML models')
    parser.add_argument('--name', type=str, required=True, 
                      choices=['gnb', 'knn', 'lr'],
                      help='Name of the model to test (gnb: GaussianNaiveBayes, knn: KNearestNeighbor)')
    
    args = parser.parse_args()
    
    try:
        if args.name == 'gnb':
            test_gaussian_naive_bayes()
        elif args.name == 'knn':
            test_knn()
        elif args.name == 'lr':
            test_lr()
        else:
            print(f"Model {args.name} not implemented yet")
    except Exception as e:
        print(f"Error testing {args.name} model: {str(e)}")

if __name__ == "__main__":
    main()
