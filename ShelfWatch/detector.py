import re
from collections import Counter, deque

import numpy as np
import supervision as sv
from ultralytics import YOLO

ALIASES = {"cerealbox": "cereal_box", "pomogranate": "pomegranate", "red_beet": "beetroot", "beet": "beetroot"}
PERSON = "person"
INVARIANT = {"asparagus", "hummus", "citrus", "brussel_sprouts", "bus", "scissors", "skis"}


def canonical(name):
    n = re.sub(r"[\s\-]+", "_", name.strip().lower())
    n = ALIASES.get(n, n)
    if n in INVARIANT:
        return n
    if n.endswith("ies"):
        n = n[:-3] + "y"
    elif n.endswith("oes"):
        n = n[:-2]
    elif n.endswith("s") and not n.endswith("ss"):
        n = n[:-1]
    return ALIASES.get(n, n)


def display(item):
    return item.replace("_", " ").capitalize()


def to_oriented(detections):
    x1, y1, x2, y2 = detections.xyxy.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1), np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)
    detections.data[sv.config.ORIENTED_BOX_COORDINATES] = corners.astype(np.float32)
    return detections


class Detector:
    def __init__(self, model_path, coco_path=None):
        self.model = YOLO(str(model_path))
        self.names = dict(self.model.names)
        self.coco = YOLO(str(coco_path)) if coco_path else None
        self.coco_offset = max(self.names) + 1
        self.coco_ids = []
        self.person_ids = []
        if self.coco:
            known = {canonical(n) for i, n in self.names.items() if self.is_valid(i)}
            for i, n in self.coco.names.items():
                self.names[self.coco_offset + i] = n
                if canonical(n) == PERSON:
                    self.person_ids.append(i)
                elif canonical(n) not in known:
                    self.coco_ids.append(i)
        self.box_annotator = sv.OrientedBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)

    def is_valid(self, class_id):
        return not self.names[class_id].startswith("class_")

    def detects_people(self):
        return bool(self.person_ids)

    def detect(self, frame, confidence, people=False, imgsz=None):
        kwargs = {"conf": confidence, "verbose": False}
        if imgsz: kwargs["imgsz"] = imgsz
        if people:
            return self.detect_coco(frame, confidence, self.person_ids, imgsz=imgsz)
        result = self.model(frame, **kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)
        if len(detections):
            valid = [self.is_valid(c) and canonical(self.names[c]) != PERSON for c in detections.class_id]
            detections = detections[np.array(valid, dtype=bool)]
        if len(detections) and sv.config.ORIENTED_BOX_COORDINATES not in detections.data:
            detections = to_oriented(detections)
        extra = self.detect_coco(frame, confidence, self.coco_ids, imgsz=imgsz)
        if len(extra) == 0:
            return detections
        return extra if len(detections) == 0 else sv.Detections.merge([detections, extra])

    def detect_coco(self, frame, confidence, class_ids, imgsz=None):
        if not self.coco or not class_ids:
            return sv.Detections.empty()
        kwargs = {"conf": confidence, "classes": class_ids, "verbose": False}
        if imgsz: kwargs["imgsz"] = imgsz
        result = self.coco(frame, **kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)
        if len(detections) == 0:
            return detections
        detections.class_id = detections.class_id + self.coco_offset
        return to_oriented(detections)

    def annotate(self, frame, detections):
        if len(detections) == 0:
            return frame.copy()
        labels = [f"{display(canonical(self.names[c]))} {p:.2f}" for c, p in zip(detections.class_id, detections.confidence)]
        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        return self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    def count(self, detections):
        return Counter(canonical(self.names[c]) for c in detections.class_id)

    def catalog(self, people=False):
        if people:
            return [PERSON] if self.person_ids else []
        return sorted({canonical(n) for i, n in self.names.items() if self.is_valid(i)} - {PERSON})


class CountSmoother:
    def __init__(self, window):
        self.history = deque(maxlen=window)

    def update(self, counts):
        self.history.append(counts)
        items = set().union(*self.history)
        return Counter({item: int(np.median([c.get(item, 0) for c in self.history])) for item in items})

    def reset(self):
        self.history.clear()


class CumulativeCounter:
    def __init__(self, detector, confidence, frame_rate):
        self.detector = detector
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            frame_rate=max(int(frame_rate), 1),
            minimum_consecutive_frames=3,
        )
        self.seen = {}

    def update(self, detections):
        tracked = self.tracker.update_with_detections(detections)
        for track_id, class_id in zip(tracked.tracker_id, tracked.class_id):
            if track_id >= 0:
                self.seen.setdefault(int(track_id), canonical(self.detector.names[class_id]))
        return Counter(self.seen.values())
