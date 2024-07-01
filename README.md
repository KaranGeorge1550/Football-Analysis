# Football Analysis

Python-based object detection and annotation of football recordings using AI/ML

## Description

This project aims to analyse and annotate a recording of a football broadcast using Machine Learning. The project is based of [this video](https://www.youtube.com/watch?v=neBZ6huolkg&list=WL&index=1&t=7842s&pp=gAQBiAQB) and I am developing this to learn about AI/ML programming and improve my Python skills.

The project utilises the [Ultralytics YOLO model](https://github.com/ultralytics/ultralytics) to automatically detect the players, ball, and referee from the video. The model was then trained on the [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) from Roboflow. In its current state, it uses this information to render a new video with custom annotations around each detection, uniquely identifying every player on the field and the ball in play. 

The next steps include automatically separating each player into their team, determining the amount of ball possession each team has, and tracking the live speed and distance ran for each player.

## Getting Started

### Dependencies

* [Python 3.12](https://www.python.org/downloads/release/python-3120/)
* [Python Virtual Environment](https://docs.python.org/3/library/venv.html)
* [Pre-trained model](https://drive.google.com/file/d/1OwIR9DcESrvF3_BmWN1EVxsovRrCdxui/view?usp=sharing)
    - MD5: `73747db8db3a4bdc6aa140559bd116f6`
* Recommend to have at least 16GB RAM

### Installing

1. Clone this project into your workspace
``` sh
git clone https://github.com/KaranGeorge1550/Football-Analysis.git
```

2. Install dependencies from [requirements.txt](/home/karan/Projects/Football-Analysis/requirements.txt) (may take a while due to Ultralytics installation)
``` sh
python3 -m venv venv && source venv/bin/activate
pip3 install -r requirements.txt
```

### Executing program

1. Add downloaded pre-trained model into `models/`

2. Add football video to analyse into `input_videos/` (recommended to be an MP4 and from a broadcast camera angle)

3. Run `main.py`
```sh
python3 main.py <video_name>
```

## Help

Please launch an issue if problem arises.

## Acknowledgments

[Code in a Jiffy](https://www.youtube.com/@codeinajiffy) for the guide.