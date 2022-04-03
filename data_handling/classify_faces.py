from recognition import face_identifier

def classify_faces(encodings, threshold=0.3):
    identifier = face_identifier.FaceIdentifier(threshold=threshold)
    identities = []
    for code in encodings:
        identity = identifier.get_identity(code)
        identities.append(identity)
    return identities, identifier