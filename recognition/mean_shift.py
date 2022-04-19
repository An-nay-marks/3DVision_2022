import numpy as np

from tqdm import tqdm
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors


class MeanShiftFaceIdentifier(MeanShift):
    def __init__(self, encoder):
        super().__init__(n_jobs=-1)
        self.encoder = encoder

    def classify_all(self, faces):
        encodings = []

        for patch in tqdm(faces):
            encodings.append(self.encoder.encode(patch))

        labels = self.fit_predict(encodings)

        nn_idx = []
        nn = NearestNeighbors(n_neighbors=min(100, len(encodings) // 4))
        nn.fit(encodings)

        for center in self.cluster_centers_:
            nbs = nn.kneighbors(np.expand_dims(center, axis=0), return_distance=False)
            nn_idx.append(nbs[0][0])

        return labels, np.asarray(nn_idx)

    @staticmethod
    def get_name(identity):
        return f'id_{identity + 1}'
