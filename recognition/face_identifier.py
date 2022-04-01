from scipy import spatial


class FaceIdentifier:
    def __init__(self, threshold):
        self.identities = []
        self.threshold = threshold

    @staticmethod
    def face_similarity(enc1, enc2):
        return 1 - spatial.distance.cosine(enc1, enc2)

    def get_identity(self, encoding):
        detected_id = -1
        max_similarity = self.threshold

        for i, id in enumerate(self.identities):
            sim = self.face_similarity(id[-1], encoding)

            if sim > max_similarity:
                detected_id = i
                max_similarity = sim

        if detected_id == -1:
            detected_id = len(self.identities)
            self.identities.append([])

        self.identities[detected_id].append(encoding)
        return detected_id
