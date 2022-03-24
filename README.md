# 3DVision_2022
3DVision project 2022 at ETH Zurich, Reconstruction Pipeline for Face Models from Videos

## How to get it running
* create Venv: `conda create --name 3DV2022 python=3.7 `
* acticate Venv: `conda activate 3DV2022`
* install dependencies: `pip install requirements.txt`
* for windows, follow the instructions in 'requirements.txt' to install dlib
* run Face Detector: `python main.py -function detect_faces --video_path ...`


## YOLOV5 steps:
* cd yolov5
* conda create --name yolov5 python=3.8
* conda activate yolov5
* pip install -qr requirements.txt
* python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
