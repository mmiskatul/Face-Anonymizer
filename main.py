import cv2
import mediapipe as mp
import argparse


def process_img(img, face_detetion):
    H, W, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detetion.process(rgb_img)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (70, 70))
    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default="video")
args.add_argument("--filePath", default='../3.mp4')

args = args.parse_args()

# detect faces
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detetion:
    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detetion)
        cv2.imwrite('data/picture1_blurred.jpeg', img)

    if args.mode in ["video"]:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        H, W, _ = frame.shape
        output_video = cv2.VideoWriter(
            'data/video_blurred.mp4',
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (W, H)
        )

        while ret:
            # process current frame
            frame = process_img(frame, face_detetion)

            # write to output video
            output_video.write(frame)

            # read next frame
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame =cv2.flip(frame,-1)
            # process current frame
            frame = process_img(frame, face_detetion)

            cv2.imshow("Webcam Face Blurring", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows() 