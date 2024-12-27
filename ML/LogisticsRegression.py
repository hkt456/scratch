import numpy as np
from dataclasses import dataclass

@dataclass
class LogisticsRegression:
    epoches: int
    learning_rate: float
    threshold: float
    logging: bool

    def sigmoid(self, predictions: np.ndarray) -> np.ndarray:

        neg_mask = predictions < 0
        pos_mask = ~neg_mask

        zs = np.empty_like(predictions)
        zs[neg_mask] = np.exp(predictions[neg_mask]) 
        zs[pos_mask] = np.exp(-predictions[pos_mask])

        res = np.ones_like(predictions) 
        res[neg_mask] = zs[neg_mask]

        return res/(1+zs)

    def mean_log_loss(self, predictions: np.ndarray, labels: np.ndarray) -> np.float32:
        return -(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)).mean()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:

        num_samples, num_features = features.shape
        self.weights, self.bias = np.zeros(num_features), 0

        for epoch in range(self.epoches):
            prediction = self.sigmoid(np.dot(features, self.weights) + self.bias)
            difference = prediction - labels

            d_weights = features.T.dot(difference) / num_samples
            d_bias = np.sum(difference) / num_samples

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

            if self.logging:
                print(f'Epoch {epoch}: loss {self.mean_log_loss(prediction, labels)}')
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.where(self.sigmoid(np.dot(features, self.weights) + self.bias) > self.threshold, 0, 1)
    