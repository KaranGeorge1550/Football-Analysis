import pickle
import cv2

def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(cache_path, width, height, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height)) # width, height
    for frame in load_frames_generator(cache_path):
        out.write(frame)
    out.release()

def load_frames_generator(cache_path):
    with open(cache_path, 'rb') as file:
        while True:
            try:
                loaded_batch = pickle.load(file)
                for frame in loaded_batch:
                    yield frame
                loaded_batch.clear()
            except EOFError:
                break