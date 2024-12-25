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
