from utils import read_video, save_video
from trackers import Tracker


def main():
    # Read video
    video_frames = read_video('input_videos/bundesliga_clip.mp4')

    # Initialise tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_path=True, 
                                       path='tracks/tracks.pkl')
    
    # Draw annotations
    print("Drawing annotations...")
    cache_path = 'tracks/frames.pkl'
    width, height = tracker.draw_annotations(video_frames, tracks, cache_path)


    # Save video
    print("Saving video...")
    output_path = 'output_videos/output_vid.avi'
    save_video(cache_path, width, height, output_path)


if __name__ == "__main__":
    main()