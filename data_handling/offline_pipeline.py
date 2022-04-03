from detect_faces import face_detection
from encode_faces import encode_faces
from classify_faces import classify_faces
from utils import get_current_datetime_as_str, ROOT_DIR
import os
import cv2

def run_offline_pipeline(video_path, model = "scrfd"):
    target_path = f"{ROOT_DIR}/data/images/extracted_patches/{get_current_datetime_as_str()}"
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    
    print("Starting with Face Detection...")
    faces = face_detection(video_path, target_path, model)
    print("...Done")
    print("Starting with Encoding faces...")
    faces, encodings = encode_faces(image_list=faces)
    print("...Done")
    print("Starting with Face Identification...")
    identities = classify_faces(encodings)
    print("...Done")
    print("Starting with DECA...")
    
    
    