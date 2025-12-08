import cv2
import mediapipe as mp
# read image
img_path = 'data/picture1.jpeg'
img =cv2.imread(img_path)

# detect faces
mp_face_detection =mp.solutions.face_detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=0) as face_detetion:
    rgb_img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out =face_detetion.process(rgb_img)
    for detection in out.detections:
        location_data = detection.location_data
        bbox = location_data.relative_bounding_box
        x1,y1,w,h =bbox.xmin, bbox.ymin, bbox.width, bbox.height



# blur faces

# save image