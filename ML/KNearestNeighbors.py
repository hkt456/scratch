import numpy as np
from dataclasses import dataclass

@dataclass
class KNearestNeighbors:
    features: np.ndarray
    labels: np.ndarray
    k: int

    