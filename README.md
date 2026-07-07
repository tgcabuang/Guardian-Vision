# Guardian-Vision
Design of a Gesture-based Video Surveillance System for Post-investigation Analysis
 - An AI-powered real-time human activity recognition and monitoring system using computer vision and deep learning.
 - 
## Overview

Guardian Vision is a desktop-based intelligent surveillance system developed as a bachelor's capstone project. It combines person detection, pose estimation, and skeleton-based action recognition to monitor human activities in almost real time.

The system is designed to assist in safety monitoring by recognizing predefined human actions through deep learning and computer vision techniques and notify personnel for violent actions.


## Dataset

Guardian Vision was trained and evaluated using the **NTU RGB+D 60 & 120** datasets, one of the largest benchmark datasets for 3D human activity recognition.

The dataset was developed by researchers from **Nanyang Technological University (NTU), Singapore**, and contains a wide variety of human action classes captured from multiple viewpoints using RGB, depth, infrared, and skeletal data.

Access to the dataset was obtained through the official request process provided by the dataset authors and was used solely for academic research and capstone development.

> **Dataset:** NTU RGB+D 60 & NTU RGB+D 120 Action Recognition Dataset  
> **Source:** https://rose1.ntu.edu.sg/dataset/actionRecognition/

**Please note:** The dataset is **not included** in this repository due to its size and the dataset's distribution policy. Researchers and developers wishing to use the dataset should request access directly from the official website.


## Screenshots of APP
<img width="914" height="516" alt="image" src="https://github.com/user-attachments/assets/81a3f8d9-4aa9-4571-a9e8-67c54749ae0a" />
<img width="920" height="575" alt="image" src="https://github.com/user-attachments/assets/cec0c093-7b57-447a-88fb-80de28afcfbd" />
<img width="919" height="517" alt="image" src="https://github.com/user-attachments/assets/2dbaa225-08ff-4e95-84d8-a2f0de3a6599" />


## Technologies

### Frontend
- HTML
- CSS
- JavaScript
- Electron

### Backend
- Python
- FastAPI (or Flask, whichever you actually use)
- OpenCV

### Artificial Intelligence
- YOLOv8
- RTMPose
- CTR-GCN


## Development Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| Node.js | 22.x |
| PyTorch | 2.1.0 |
| CUDA | 11.8 |
| MMCV | 2.1.0 |



## Installation

### Clone the repository

git clone https://github.com/yourusername/Guardian-Vision.git
cd Guardian-Vision

npm install
pip install -r requirements.txt
pip install -U openmim
mim install mmcv

For additional installation instructions, refer to the official MMCV documentation:
https://mmcv.readthedocs.io/en/latest/get_started/installation.html



| MMDetection | 3.2.0 |
| MMPose | 1.3.2 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
