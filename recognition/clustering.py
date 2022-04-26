import numpy as np

from tqdm import tqdm
from sklearn import base, cluster, neighbors


class _ClusterFaceClassifier(base.ClusterMixin):
    def __init__(self, encoder):
        self.encoder = encoder
        self.encodings = None
    
    def encode_all(self, faces):
        encodings = []
        for patch in tqdm(faces):
            encodings.append(self.encoder.encode(patch))
        self.encodings = encodings
        return encodings
            
    def classify_all(self, faces):
        if self.encodings is None:
            self.encode_all(faces)
        encodings = self.encodings
        self.encodings = None # in case the same classifier object is being used for classification of different patches
        return encodings, self.fit_predict(encodings)

    @staticmethod
    def get_name(identity):
        return 'Unknown' if identity == -1 else f'id_{identity + 1}'
    

class MeanShiftFaceClassifier(_ClusterFaceClassifier, cluster.MeanShift):
    def __init__(self, encoder):
        super().__init__(encoder)
        super(_ClusterFaceClassifier, self).__init__(n_jobs=-1)

    def classify_all(self, faces):
        encodings, labels = super().classify_all(faces)

        nn_idx = []
        nn = neighbors.NearestNeighbors(n_neighbors=min(100, len(encodings) // 4))
        nn.fit(encodings)

        for center in self.cluster_centers_:
            nbs = nn.kneighbors(np.expand_dims(center, axis=0), return_distance=False)
            nn_idx.append(nbs[0][0])

        return labels, np.asarray(nn_idx)


class AgglomerativeFaceClassifier(_ClusterFaceClassifier, cluster.AgglomerativeClustering):
    def __init__(self, encoder, threshold):
        super().__init__(encoder)
        super(_ClusterFaceClassifier, self).__init__(
            None, affinity='cosine', linkage='average', distance_threshold=1-threshold)

    def classify_all(self, faces):
        _, labels = super().classify_all(faces)
        return labels, np.arange(labels.size)


class DBSCANFaceClassifier(_ClusterFaceClassifier, cluster.OPTICS):
    def __init__(self, encoder, threshold):
        super().__init__(encoder)
        super(_ClusterFaceClassifier, self).__init__(metric='cosine', eps=1-threshold, n_jobs=-1)

    def classify_all(self, faces):
        _, labels = super().classify_all(faces)
        return labels, np.arange(labels.size)
