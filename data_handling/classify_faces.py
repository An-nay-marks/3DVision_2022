from recognition import face_identifier

def classify_faces(encodings, threshold=0.3):
    """Face classification/identification

    Args:
        encodings (list): encodings created by the encoder in the previous step in the pipeline
        threshold (float, optional): Similarity threshold for comparing two images. Defaults to 0.3.

    Returns:
        tuple(list, list): returns identities in order of encodings and identifier
    """
    identifier = face_identifier.FaceIdentifier(threshold=threshold)
    identities = []
    for code in encodings:
        identity = identifier.get_identity(code)
        identities.append(identity)
    return (identities, identifier)