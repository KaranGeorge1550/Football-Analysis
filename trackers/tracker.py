from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
    
    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

         #Overwrite goalkeeper class with player class
        for frame, detection in enumerate(detections):
            class_names = detections.names
            class_names_inverse = {v: k for k, v in class_names.items()}

            # To supervision detection
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Goalkeeper to player
            for object, class_id in detection_supervision.classes.items():
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object] = class_names_inverse['player']