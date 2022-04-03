import copy
import sys
import cv2
import numpy as np
import torch

sys.path.append('yolo5_face')
from detection.yolo5_face import detect_face as yolo5


class YOLOv5FaceDetector:
    def __init__(self, model_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = yolo5.load_model(model_file, self.device)

    def detect(self, frame):
        img_size = 800
        conf_thres = 0.5
        iou_thres = 0.5

        img0 = copy.deepcopy(frame)
        h0, w0 = frame.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = yolo5.check_img_size(img_size, s=self.model.stride.max())  # check img_size

        img = yolo5.letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = yolo5.non_max_suppression_face(pred, conf_thres, iou_thres)
        faces = []

        for i, det in enumerate(pred):  # detections per image
            # gn = torch.tensor(frame.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain
            # gn_lks = torch.tensor(frame.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = yolo5.scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # det[:, 5:15] = yolo5.scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                for j in range(det.size()[0]):
                    face = det[j, :5].cpu().numpy()
                    # landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    faces.append(face)

        return np.asarray(faces)
