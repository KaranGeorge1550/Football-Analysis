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
    
    def get_object_tracks(self, frame_num):
        detections = self.detect_frames(frame_num)

        tracks={
            "players": [],
            "ball": [],
            "referees": [],
        }

         #Overwrite goalkeeper class with player class
        for frame_num, detection in enumerate(detections):
            class_names = detections.names
            class_names_inverse = {v: k for k, v in class_names.items()}

            # To supervision detection
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Goalkeeper to player
            for object, class_id in detection_supervision.classes.items():
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object] = class_names_inverse['player']

            # Track
            detection_tracking = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for frame_num in detection_tracking:
                bounding_box = frame_num[0].tolist()
                class_id = frame_num[3]
                tracker_id = frame_num[4]

                if class_id == class_names_inverse['player']:
                    tracks["players"][frame_num][tracker_id] = {"bounding_box":bounding_box}
                elif class_id == class_names_inverse['ball']:
                    tracks["ball"][frame_num][1] = {"bounding_box":bounding_box}
                elif class_id == class_names_inverse['referee']:
                    tracks["referees"][frame_num][tracker_id] = {"bounding_box":bounding_box} 

        return tracks 
