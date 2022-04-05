from typing import List
from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm

from sklearn.cluster import MeanShift


@dataclass
class Identity:
    num_encodings: int = 0
    mean_encoding: np.array = 0

    def add_encoding(self, enc):
        n = self.num_encodings + 1
        self.mean_encoding = ((n - 1) * self.mean_encoding + enc) / n
        self.num_encodings = n


class RealTimeFaceIdentifier:
    def __init__(self, threshold):
        self.identities: List[Identity] = []
        self.threshold: float = threshold

    @staticmethod
    def face_similarity(enc1, enc2):
        return enc1 @ enc2 / (norm(enc1) * norm(enc2))

    def get_identity(self, encoding):
        detected_id = -1
        max_similarity = self.threshold

        for i, identity in enumerate(self.identities):
            sim = self.face_similarity(encoding, identity.mean_encoding)

            if sim > max_similarity:
                detected_id = i
                max_similarity = sim

        if detected_id == -1:
            detected_id = len(self.identities)
            self.identities.append(Identity())

        self.identities[detected_id].add_encoding(encoding)
        return detected_id


class OfflineFaceIdentifier:
    """Uses Mean Shift to get clusters of identities, based on the facial encoding
    """
    
    def get_identities(self, encodings):
        """performs mean shift

        Args:
            encodings (list): encodings as a list of codes

        Returns:
            tuple(list, list, list): 
        """
        ms = MeanShift()
        labels = ms.fit_predict(encodings)
        labels_unique = np.unique(labels)
        cluster_centers = ms.cluster_centers_
        return labels_unique.tolist(), cluster_centers.tolist(), labels.tolist()