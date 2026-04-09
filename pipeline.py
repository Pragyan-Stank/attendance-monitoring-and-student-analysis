import cv2
import numpy as np
import insightface
import database as db
import threading
import logging
import base64
import time
import torch
import torch.backends.cudnn as cudnn
import onnxruntime as ort

# Optimize pyTorch for GPU
cudnn.benchmark = True

class AsyncPipeline:
    def __init__(self):
        self.face_app = None
        self.student_db = {}
        self.session_id = None
        self.attendance_marked = set()
        self._initialized = False
        self._lock = threading.Lock()
        
        # Tracking & Caching
        self.track_cache = {} # track_id -> {student_id, name, conf, last_rec_time, bbox}
        self.next_track_id = 0

        self.frame_count = 0
        
        # Architectural Rules
        self.DETECTION_INTERVAL =5    #Run detection every 4 frames
        self.RECOGNITION_COOLDOWN = 3.0 # sec
        self.DROPOUT_COOLDOWN = 2.0 # drop tracker if lost for 1 sec

    def initialize(self):
        with self._lock:
            if self._initialized: return
            logging.getLogger('insightface').setLevel(logging.ERROR)
            
            # 8. Optimize GPU usage
            available = ort.get_available_providers()
            providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

            print("--- INITIALIZING HIGH-FPS TRACKING PIPELINE (KCF + ArcFace) ---")
            self.face_app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=['detection', 'recognition'] 
            )
            
            # Using 320x320 for rapid detector
            self.face_app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.45) 
            self._initialized = True
            print("--- PIPELINE READY ---")

            self.load_student_db()

    def load_student_db(self):
        self.student_db = {
            r["student_id"]: {"name": r["name"], "embedding": r["embedding"]}
            for r in db.get_all_embeddings()
        }

    def start_session(self, session_id: str):
        self.session_id = session_id
        self.attendance_marked = set()
        self.track_cache.clear()
        self.frame_count = 0
        self.load_student_db()


    def identify_face(self, embedding: np.ndarray) -> tuple[str, float]:
        if not self.student_db: return None, 0.0
        
        best_id, best_score = None, -1.0
        for sid, data in self.student_db.items():
            score = float(np.dot(embedding, data["embedding"]))
            if score > best_score:
                best_score, best_id = score, sid
        
        if best_score >= 0.45:
            return best_id, best_score
        return None, best_score

    def capture_enrollment(self, frame_b64) -> tuple[np.ndarray, np.ndarray]:
        if not self._initialized: self.initialize()
        img_data = base64.b64decode(frame_b64)
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        self.face_app.det_model.input_size = (640, 640)
        faces = self.face_app.get(frame)
        self.face_app.det_model.input_size = (320, 320)
        
        if not faces:
            return None, frame
            
        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        if best_face.normed_embedding is None:
            return None, frame
            
        return best_face.normed_embedding, frame

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def process_frame(self, frame_b64) -> list:
        if not self._initialized: self.initialize()
        
        img_data = base64.b64decode(frame_b64)
        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        # 7. Downscale input frames
        orig_h, orig_w = frame.shape[:2]
        target_w, target_h = 640, 360
        frame = cv2.resize(frame, (target_w, target_h))
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        
        self.frame_count += 1
        current_time = time.time()
        
        # --- DROPOUT PHASE ---
        lost_tracks = []
        for track_id, cache in self.track_cache.items():
            if current_time - cache["last_seen"] > self.DROPOUT_COOLDOWN:
                lost_tracks.append(track_id)
                
        for t_id in lost_tracks:
            del self.track_cache[t_id]

        # --- DETECTION & TRACKING PHASE (Runs Every N Frames) ---
        if self.frame_count % self.DETECTION_INTERVAL == 0:
            faces = self.face_app.get(frame)

            
            for face in faces:
                bbox = face.bbox.tolist()
                
                # Match detection to existing tracks
                best_iou = 0
                best_track_id = None
                for track_id, cache in self.track_cache.items():
                    iou = self.calculate_iou(bbox, cache["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
                
                # If no match or low overlap, register new track
                if best_iou < 0.35:
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    
                    self.track_cache[track_id] = {
                        "student_id": None,
                        "name": "Unknown",
                        "confidence": 0.0,
                        "last_rec_time": 0.0,
                        "last_seen": current_time,
                        "bbox": bbox
                    }
                    best_track_id = track_id
                else:
                    # Update bounding box of existing track to align with NN
                    self.track_cache[best_track_id]["bbox"] = bbox
                    self.track_cache[best_track_id]["last_seen"] = current_time

                # --- RECOGNITION PHASE (Runs conditionally) ---

                cache = self.track_cache[best_track_id]
                should_recognize = (cache["student_id"] is None) and (current_time - cache["last_rec_time"] > self.RECOGNITION_COOLDOWN)
                
                if should_recognize and face.normed_embedding is not None:
                    cache["last_rec_time"] = current_time
                    sid, conf = self.identify_face(face.normed_embedding)
                    
                    if sid:
                        cache["student_id"] = sid
                        cache["name"] = self.student_db[sid]["name"]
                        cache["confidence"] = conf
                        
                        # Logging Attendance
                        if self.session_id and sid not in self.attendance_marked:
                            db.mark_attendance(self.session_id, sid, conf)
                            self.attendance_marked.add(sid)


        # Build final results mapped to original dimensions
        results = []
        for track_id, cache in self.track_cache.items():
            b = cache["bbox"]
            scaled_bbox = [b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y]
            results.append({
                "student_id": cache["student_id"],
                "name": cache["name"],
                "track_id": track_id,
                "bbox": scaled_bbox,
                "present": cache["student_id"] is not None,
                "confidence": round(cache["confidence"], 3),
                "is_focused": True,
                "distraction_type": None,
                "engagement_score": 1.0
            })
            
        return results
