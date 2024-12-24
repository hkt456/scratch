import numpy as np

class GaussianNaiveBayes:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits the model to the data."""

        self.labels = labels
        self.unique_labels = np.unique(labels)

        self.params = []

        for label in self.unique_labels:
            label_features = features[self.labels == label]
            self.params.append([(col.mean(), col.var()) for col in label_features.T])
        
    def likelihood(self, data: float, mean: float, var: float) -> float:
        eps = 1e-4
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((data - mean) ** 2 / (2 * var + eps)))
        return coeff * exponent


    def predict(self, features: np.ndarray) -> np.ndarray:
        """Perfoms inference using Bayes' theorem: P(y|x) = P(x|y)P(y) / P(x)"""
        num_samples, _ = features.shape

        predictions = np.empty(num_samples)
        
        for i, feature in enumerate(features):
            posteriors = []
            for label_index, label in enumerate(self.unique_labels):
                prior = np.log((self.labels == label).mean())

                """
                Naive assumption:
                P(x|y) = P(x1|y)P(x2|y)...P(xn|y)
                """

                pairs = zip(feature, self.params[label_index])
                likelihood = np.sum([np.log(self.likelihood(f, m, v)) for f, (m, v) in pairs])

                posteriors.append(prior + likelihood)
        
            predictions[i] = self.unique_labels[np.argmax(posteriors)]

        return predictions

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    features, labels = load_iris(return_X_y=True)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.5, random_state=0)

    gnb = GaussianNaiveBayes()
    gnb.fit(train_features, train_labels)
    predictions = gnb.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, predictions, average="macro")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {fscore}")
    print()
    print(f"Mislabeled points: {(predictions != test_labels).sum()}/{test_features.shape[0]}")