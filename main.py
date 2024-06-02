from utils import read_video, save_video
from trackers import Tracker


def main():
    video_frames = read_video('input_videos/hockey_input1.mp4')

    # Initialise tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_path=True, 
                                       path='tracks/tracks.pkl')

    save_video(video_frames, 'output_videos/output_vid.avi')

if __name__ == "__main__":
    main()