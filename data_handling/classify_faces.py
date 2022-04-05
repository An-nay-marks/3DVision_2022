import numpy as np
from recognition import face_identifier, vgg
import cv2
from sklearn.neighbors import NearestNeighbors

def classify_faces_online(encodings, threshold=0.3):
    """Face classification/identification

    Args:
        encodings (list): encodings created by the encoder in the previous step in the pipeline
        threshold (float, optional): Similarity threshold for comparing two images. Defaults to 0.3.

    Returns:
        tuple(list, RealTimeFaceIdentifier): returns identities in order of encodings and identifier
    """
    identifier = face_identifier.RealTimeFaceIdentifier(threshold=threshold)
    identities = []
    for code in encodings:
        identity = identifier.get_identity(code)
        identities.append(identity)
    return identities, identifier

def classify_meanshift(encodings):
    identifier = face_identifier.OfflineFaceIdentifier()
    unique_labels, cluster_centers, identities = identifier.get_identities(encodings)
    return unique_labels, cluster_centers, identities

def classify_vgg(images, threshold=0.3):
    """Face classification/identification

    Args:
        images (list): list of face patches, can be of different sizes, because they will be rescaled
        threshold (float, optional): Similarity Threshold. Defaults to 0.3.

    Returns:
        list: identities in order of encodings
    """
    # Example of face detection with a vggface2 model
    classifications = []
    classifier = vgg.VGGEncoderAndClassifier(threshold)
    
    for img in images:
        # highest scored name and probability
        classifications.append(classifier.classify(img))
    return classifications

def getNN(cluster_centers, encodings):
    NN_indexs = []
    cluster_centers = np.array(cluster_centers)
    encods = np.array(encodings)
    nei = NearestNeighbors(n_neighbors=min(100, len(encodings)//4))
    nei.fit(encods)
    for center in cluster_centers:
        neighbs = nei.kneighbors(np.expand_dims(center, axis = 0), return_distance=False)
        NN_indexs.append(neighbs[0][0])
    return NN_indexs