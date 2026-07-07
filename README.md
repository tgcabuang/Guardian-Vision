# Guardian-Vision
Design of a Gesture-based Video Surveillance System for Post-investigation Analysis
 - An AI-powered real-time human activity recognition and monitoring system using computer vision and deep learning.


## Overview

Guardian Vision is a desktop-based intelligent surveillance system developed as a bachelor's capstone project. It combines person detection, pose estimation, and skeleton-based action recognition to monitor human activities in almost real time.

The system is designed to assist in safety monitoring by recognizing predefined human actions through deep learning and computer vision techniques and notify personnel for violent actions.

## Installation

For additional installation instructions, refer to the official MMCV documentation:
https://mmcv.readthedocs.io/en/latest/get_started/installation.html

## Application

### Clone the repository

```bash
git clone https://github.com/yourusername/Guardian-Vision.git
cd Guardian-Vision

npm install
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
```


## Dataset

Guardian Vision was trained and evaluated using the **NTU RGB+D 60 & 120** datasets, one of the largest benchmark datasets for 3D human activity recognition.

The dataset was developed by researchers from **Nanyang Technological University (NTU), Singapore**, and contains a wide variety of human action classes captured from multiple viewpoints using RGB, depth, infrared, and skeletal data.

Access to the dataset was obtained through the official request process provided by the dataset authors and was used solely for academic research and capstone development.

> **Dataset:** NTU RGB+D 60 & NTU RGB+D 120 Action Recognition Dataset  
> **Source:** https://rose1.ntu.edu.sg/dataset/actionRecognition/

**Please note:** The dataset is **not included** in this repository due to its size and the dataset's distribution policy. Researchers and developers wishing to use the dataset should request access directly from the official website.


## Model Weights

Trained model weights (`.pth` and `.pt`) are **not included** in this repository.

After training the models using the scripts in the `builder_trainer/` directory, place the generated weight files in the `backend/assets` directory before running the application.

Required model files include, but are not limited to:

- `ctrgcn_bundle.pth`
- `rtmpose.pth`
- `yolov8n.pt`

Please refer to the project documentation or configuration files for the expected file locations.


## Required Pretrained Models

### RTMPose

This project uses **RTMPose-S** (COCO, 17 keypoints, 256×192 input).

Download the official pretrained checkpoint from the MMPose Model Zoo:

- https://github.com/open-mmlab/mmpose
- https://mmpose.readthedocs.io/en/latest/model_zoo.html

After downloading, rename the checkpoint to:

```text
rtmpose.pth
```

Place it in:

```text
backend/assets/rtmpose.pth
```


# Dataset Builder and Model Training

## Prerequisites

This repository **does not include** the NTU RGB+D 60 & 120 datasets.

Before training the model, you must first obtain the dataset by requesting access through the official website:

https://rose1.ntu.edu.sg/dataset/actionRecognition/

Please ensure that your use of the dataset complies with the dataset's terms and conditions.

---

## Preparing the Dataset

After downloading the NTU RGB+D dataset:

1. Extract the dataset.
2. Copy the required video files into the dataset directory used by the dataset builder.
3. **Do not rename the videos.**

The dataset builder expects the **original NTU RGB+D filenames**, for example:

```text
S001C001P001R001A001.mp4
S001C001P001R001A002.mp4
S001C001P001R001A003.mp4
...
```

These filenames contain metadata used during preprocessing and training.

---

## Building the Dataset

Run the dataset builder to preprocess the videos and generate the required training files.

```bash
python builder_trainer /dataset_builder_combine_recommended.py
```

The dataset builder will:

- Read the NTU RGB+D videos
- Extract the required information
- Generate the dataset format required for training

---


## Training the Model

After the dataset has been prepared, train the action recognition model:

```bash
python builder_trainer/training_with_cctv_aug.py
```

Training will generate the model weights (`.pth`) that are required by Guardian Vision.

---

## Using the Trained Model

After training is complete, place the generated model weights in the `backend/assets` directory used by the application.

The application will automatically load the trained model during inference.

---

## Notes

- The NTU RGB+D dataset is **not distributed** with this repository.
- Users must obtain the dataset directly from the dataset authors.
- The original dataset filenames should be preserved during preprocessing.
- This repository provides only the tools required to prepare the dataset and train the model.


## Screenshots of APP
<img width="1341" height="718" alt="image" src="https://github.com/user-attachments/assets/79196baf-a7a9-4c0d-8b71-d4f8cd0d5d1a" />
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
- FastAPI
- OpenCV

### Artificial Intelligence
- YOLOv8
- RTMPose
- CTR-GCN

### App Compilation
- Inno Setup


## Development Environment

| Component | Version |
|-----------|---------|
| Python | 3.9 |
| Node.js | 22.x |
| PyTorch | 2.1.0 |
| CUDA | 11.8 |
| MMCV | 2.1.0 |
| MMDetection | 3.2.0 |
| MMPose | 1.3.2 |

## Laptop Specs Used and Its FPS:
| CPU | Ryzen 7 5800H with Radeon Graphics|
|-----------|---------|
| GPU | NVIDIA GeForce RTX 3050 Ti |

| 1 CAM | 2 CAMS | 3 CAMS |
|-----------|---------|---------|
| 15 Fps | 10 Fps | 5 Fps |




