import cv2
from mtcnn import MTCNN

class Model1MTCNN:
    def __init__(self):
        print("Loading MTCNN...")
        self.detector = MTCNN()
        print("MTCNN loaded.")

    # Only detect faces
    def detect(self, frame):
        try:
            detections = self.detector.detect_faces(frame)
        except Exception:
            detections = []
        return detections

    # Draw bounding boxes + keypoints
    def draw(self, frame, detections):
        for face in detections:
            if 'box' not in face:
                continue

            x, y, w, h = face['box']

            if w <= 0 or h <= 0:
                continue

            cv2.rectangle(frame, (x, y),
                          (x + w, y + h),
                          (255, 0, 0), 2)

            if 'keypoints' in face:
                for point in face['keypoints'].values():
                    cv2.circle(frame, point,
                               4, (0, 255, 0), -1)

        return frame
