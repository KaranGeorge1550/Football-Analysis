from ultralytics import YOLO

model = YOLO('yolov8l')

results = model.predict('input_videos/hockey_input1.mp4', save=True)
print(results[0])
print('-----------------------------------')
for box in results[0].boxes:
    print(box)