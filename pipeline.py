import cv2
import numpy as np
import insightface
import database as db
import threading
import logging
import base64
import onnxruntime as ort

class FastPipeline:
    def __init__(self):
        self.face_app = None
        self.student_db = {}
        self.session_id = None
        self.attendance_marked = set()
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self):
        with self._lock:
            if self._initialized: return
            logging.getLogger('insightface').setLevel(logging.ERROR)
            
            # Prioritize execution speed
            available = ort.get_available_providers()
            providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

            print("--- INITIALIZING LIGHTWEIGHT PIPELINE ---")
            # BY RESTRICTING ALLOWED MODULES, WE SKIP LANDMARKS & AGE/GENDER (Massive CPU Savings)
            self.face_app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=['detection', 'recognition'] 
            )
            
            # 320x320 is extremely fast and acts like YOLO Nano
            self.face_app.prepare(ctx_id=-1, det_size=(320, 320)) 
            self._initialized = True
            print("--- LIGHTWEIGHT PIPELINE READY ---")

    def load_student_db(self):
        self.student_db = {
            r["student_id"]: {"name": r["name"], "embedding": r["embedding"]}
            for r in db.get_all_embeddings()
        }

    def start_session(self, session_id: str):
        self.session_id = session_id
        self.attendance_marked = set()
        self.load_student_db()

    def identify_face(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.student_db: return None, 0.0
        
        # Fast dot product matrix calculation to find max match
        best_id, best_score = None, -1.0
        for sid, data in self.student_db.items():
            score = float(np.dot(embedding, data["embedding"]))
            if score > best_score:
                best_score, best_id = score, sid
        
        if best_score >= 0.45: # slightly relaxed threshold for low-res crops
            return best_id, best_score
        return None, best_score

    def capture_enrollment(self, frame_b64) -> tuple[np.ndarray, np.ndarray]:
        if not self._initialized: self.initialize()
        img_data = base64.b64decode(frame_b64)
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Increase res just for enrollment to ensure perfect descriptor
        self.face_app.det_model.input_size = (640, 640)
        faces = self.face_app.get(frame)
        self.face_app.det_model.input_size = (320, 320) # Reset to fast
        
        if not faces:
            return None, frame
            
        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        if best_face.normed_embedding is None:
            return None, frame
            
        return best_face.normed_embedding, frame

    def process_frame(self, frame_b64) -> list:
        if not self._initialized: self.initialize()
        
        img_data = base64.b64decode(frame_b64)
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        results = []

        # High speed face detection
        faces = self.face_app.get(frame)
        if not faces: return []

        for face in faces:
            bbox = face.bbox.tolist()
            student_id, confidence = None, 0.0
            
            # Recognition
            if face.normed_embedding is not None:
                student_id, confidence = self.identify_face(face.normed_embedding)
            
            # Logging Attendance if recognized
            if student_id and self.session_id and student_id not in self.attendance_marked:
                db.mark_attendance(self.session_id, student_id, confidence)
                self.attendance_marked.add(student_id)

            name = self.student_db[student_id]["name"] if student_id else "Unknown"

            results.append({
                "student_id": student_id,
                "name": name,
                "track_id": student_id or "unknown",  # Bypass tracker, use raw ID 
                "bbox": bbox,
                "present": student_id is not None,
                "confidence": round(confidence, 3),
                "is_focused": True, # Placeholder
                "distraction_type": None, # Features disabled
                "engagement_score": 1.0 # Features disabled
            })

        return results
