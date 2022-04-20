import numpy as np

from typing import List
from numpy.linalg import norm


class Identity:
    num_encodings: int = 0
    mean_encoding: np.array = 0

    def add_encoding(self, enc):
        n = self.num_encodings + 1
        self.mean_encoding = ((n - 1) * self.mean_encoding + enc) / n
        self.num_encodings = n


class RealTimeFaceIdentifier:
    def __init__(self, encoder, threshold):
        self.identities: List[Identity] = []
        self.encoder = encoder
        self.threshold: float = threshold

    def classify(self, img):
        detected_id = -1
        max_similarity = self.threshold
        encoding = self.encoder.encode(img)

        for i, identity in enumerate(self.identities):
            sim = self.encoder.similarity(encoding, identity.mean_encoding)

            if sim > max_similarity:
                detected_id = i
                max_similarity = sim

        if detected_id == -1:
            detected_id = len(self.identities)
            self.identities.append(Identity())

        self.identities[detected_id].add_encoding(encoding)
        return detected_id

    @staticmethod
    def get_name(identity):
        return f'id_{identity + 1}'

    def get_num_samples(self, identity):
        if identity >= len(self.identities):
            return 0

        return self.identities[identity].num_encodings
