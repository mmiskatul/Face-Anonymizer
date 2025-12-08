import cv2
import mediapipe as mp
# read image
img_path = 'data/picture1.jpeg'
img =cv2.imread(img_path)

# detect faces
mp_face_detection =mp.solutions.face_detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=0) as face_detetion:
    pass

# blur faces

# save image