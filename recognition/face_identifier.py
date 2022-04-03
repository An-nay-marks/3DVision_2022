from typing import List
from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm


@dataclass
class Identity:
    num_encodings: int = 0
    mean_encoding: np.array = np.zeros(512)

    def add_encoding(self, enc):
        n = self.num_encodings + 1
        self.mean_encoding = ((n - 1) * self.mean_encoding + enc) / n
        self.num_encodings = n


class FaceIdentifier:
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
