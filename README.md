# Keypoint Extraction using OpenPose

This project uses OpenPose to extract keypoints from videos. It provides functionalities to visualize these keypoints by drawing skeletons on videos and exporting the keypoints data to both `.npy` and `.mat` formats.

## Features

- Extract keypoints from video using OpenPose.
- Generate videos with overlaid skeletons to visualize keypoints.
- Export keypoints data to `.npy` (Numpy array format) and `.mat` (MATLAB format).


## Requirements

- Nvidia Docker runtime: https://github.com/NVIDIA/nvidia-docker#quickstart
- CUDA 10.0 or higher on your host, check with nvidia-smi


## Prerequisites

- Python 3.6
- Docker image https://hub.docker.com/r/cwaffles/openpose
## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/JindrichA/openpose_extractor.git
   cd openpose_extractor
   ```

