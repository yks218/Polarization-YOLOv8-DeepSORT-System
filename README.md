Polarization-Aware Object Detection and Tracking System
YOLOv8/YOLOv11 + DeepSORT + pBRDF Modeling
Project Overview

This project develops a polarization-aware intelligent object detection and tracking system by integrating physics-based optical modeling and deep learning algorithms.
The system combines:
Polarization imaging theory
pBRDF (polarized Bidirectional Reflectance Distribution Function) modeling
Hyperspectral band optimization
YOLO-based object detection
DeepSORT multi-object tracking
Tkinter interactive graphical user interface (GUI)

The proposed system aims to improve object perception capability under complex imaging conditions, including:

Low contrast scenes
Scattering environments
Spectral imaging scenarios
Multi-object monitoring scenarios

A Python-based YOLO + DeepSORT system with GUI visualization for image and video object detection.

Key Features
1. YOLO-Based Object Detection
Implemented functions:
Single image detection
Batch image detection
Custom dataset training
Model weight replacement
Detection result visualization
2. Video Detection and Tracking
The system integrates YOLO detection with DeepSORT tracking.
Features:
Video file import
Real-time pedestrian detection
Multi-object tracking
ID assignment
Trajectory visualization
Object counting
GUI functions:
Function	Description
Video Import	Load video files
Pedestrian Detection	YOLO-based detection with bounding boxes
Pedestrian Tracking	DeepSORT tracking with trajectories
Pause/Play	Control video playback
FPS display	Real-time processing speed
Count	Current number of objects
Total	Total tracked IDs
In/Out	Object crossing statistics
3. pBRDF-Based Polarization Modeling
A physics-based polarized Bidirectional Reflectance Distribution Function (pBRDF) model is implemented in MATLAB.
The model considers:
Incident angle
Observation angle
Surface roughness
Refractive index
Extinction coefficient
The input optical parameters include:
Wavelength
        |
        |
Refractive index n(λ)
Extinction coefficient k(λ)
Surface roughness
Camera response curve
Solar spectrum
Polarizer transmission curve
        ↓
pBRDF Simulation
         ↓
Degree of Polarization (DoP)
        ↓
Optimal hyperspectral band selection

4. GUI Software Platform
The interactive interface is developed using Tkinter.
The software supports:
Image Processing
Image folder import
YOLO object detection
Detection result visualization
Sequential image browsing
Batch detection overview
Video Processing
Video import
Pedestrian detection
Pedestrian trajectory tracking
Pause/play control
The GUI provides an easy-to-use interface without requiring command-line operation.

Project Structure
Polarization-YOLOv8-Detection/
│
├── ui-main.py                  # Main GUI entrance
├── ui.py                       # Tkinter interface
├── video_tracker.py            # YOLO + DeepSORT video tracking
│
├── detect.py                   # Image detection
├── tracker.py                  # Tracking related functions
│
├── main_yolov8.py              # Original YOLOv8 + DeepSORT demo
├── main_yolov11.py             # YOLOv11 demo
├── image2video.py              # Image sequence to video
│
├── export_onnx.py              # Export YOLO model to ONNX
├── ONNX.py                     # ONNX inference test
│
├── deep_sort/                  # DeepSORT implementation
│   ├── deep/
│   ├── configs/
│   └── checkpoint/
│
├── weights/                    # YOLO model weights
│   ├── yolov8n.pt
│   ├── yolov8l.pt
│   ├── yolov8.pt
│   └── yolov11n.pt
│
├── yolov8n_cbam.yaml           # Custom YOLOv8-CBAM structure
│
├── main_pbrdf.m                # pBRDF main program
├── compute_pbrdf.m             # pBRDF calculation
├── config.m                    # pBRDF parameter configuration
│
├── data/                       # MATLAB input parameters
│
├── images/                     # Training/testing images
│
├── labels/                     # YOLO annotation files
│
├── icon/                       # GUI icons
│
├── output_sample/              # Example results and screenshots
│
├── requirements.txt
├── LICENSE
└── README.md

Installation
Recommended:
Python >= 3.8
CUDA >= 11.x
Install dependencies:
pip install -r requirements.txt