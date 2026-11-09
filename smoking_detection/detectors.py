from pathlib import Path
import torch
from ultralytics import YOLO

COCO_PERSON_CLASS_ID = 0

class PersonDetector:
    def __init__(self, weights_path="yolo26s.pt", conf=0.35, iou=0.45, device="auto"):
        if not Path(weights_path).exists() and "/" in str(weights_path):
            raise FileNotFoundError(f"Person model weights not found: {weights_path}")
        self.device = self._resolve_device(device)
        print(f"[INFO] Loading person detector: {weights_path} on {self.device}")
        self.model = YOLO(weights_path)
        self.conf, self.iou = conf, iou

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        return device

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, iou=self.iou,
            classes=[COCO_PERSON_CLASS_ID], device=self.device, verbose=False)
        persons = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                score = float(box.conf[0].cpu().item())
                if x2 > x1 and y2 > y1:
                    persons.append({"box": (x1,y1,x2,y2), "conf": score})
        return persons

class SmokingDetector:
    """Runs a custom YOLO detector on each person crop.

    The custom best_smoke.pt shown in the video is not embedded in the recording.
    Class names are read from the checkpoint when available.
    """
    def __init__(self, weights_path, conf=0.25, iou=0.45, device="cpu"):
        path = Path(weights_path)
        self.enabled = path.exists()
        self.model = None
        self.conf, self.iou, self.device = conf, iou, device
        self.names = {}
        if self.enabled:
            print(f"[INFO] Loading smoking detector: {path}")
            self.model = YOLO(str(path))
            self.names = self.model.names
        else:
            print(f"[WARN] Smoking weights not found: {path}")
            print("[WARN] Person detection will run, but smoking classification is disabled.")
            print("[WARN] Add the trained best_smoke.pt checkpoint to enable smoking detection.")

    def detect_in_crop(self, crop):
        if crop is None or crop.size == 0 or not self.enabled:
            return []
        results = self.model.predict(crop, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().item())
                score = float(box.conf[0].cpu().item())
                label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else self.names[cls_id]
                detections.append({"box":(x1,y1,x2,y2),"class_id":cls_id,
                                   "label":str(label),"conf":score})
        return detections
