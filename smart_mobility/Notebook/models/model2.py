import cv2
import numpy as np
import onnxruntime as ort

FACE_SIZE = (112, 112)
THRESHOLD = 0.6

def l2_norm(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)


def preprocess(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, FACE_SIZE)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.expand_dims(face, axis=0)
    return face

def align_face(img, detection):
    keypoints = detection['keypoints']

    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, rot_mat, img.shape[1::-1])

    x, y, w, h = detection['box']
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = x1 + w
    y2 = y1 + h

    face = aligned[y1:y2, x1:x2]

    return face

class Model2ArcFace:
    def __init__(self):
        print("Loading ArcFace model...")
        self.session = ort.InferenceSession(
            "models/arcface.onnx",
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.db = np.load(
            "models/arcface_db.npy",
            allow_pickle=True
        ).item()

        print("ArcFace ready.")

    def get_embedding(self, face):
        x = preprocess(face)
        emb = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )[0]
        emb = emb.reshape(-1)
        return l2_norm(emb)

    def recognize(self, emb):
        best_name = "Unknown"
        best_score = -1

        for name, db_emb in self.db.items():
            score = np.dot(emb, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < THRESHOLD:
            return "Unknown", best_score

        return best_name, best_score

    def process(self, frame, detections):
        for d in detections:

            if 'box' not in d or 'keypoints' not in d:
                continue

            x, y, w, h = d['box']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = x1 + w
            y2 = y1 + h

            face = align_face(frame, d)

            if face is None:
                continue

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            emb = self.get_embedding(face)
            name, score = self.recognize(emb)

            label = f"{name} | {score:.2f}"

            cv2.putText(frame, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255), 2)

        return frame
