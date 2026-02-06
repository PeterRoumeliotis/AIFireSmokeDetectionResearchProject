# AIFireSmokeDetectionResearchProject

 Overview                               

  This research project explores deep learning approaches for detecting fire in
  images and video frames. It implements and compares two distinct computer
  vision architectures — a custom Convolutional Neural Network (CNN) built with
  TensorFlow/Keras and a fine-tuned YOLOv8 Nano object detection model — to
  evaluate their effectiveness for real-time wildfire detection.

  Models

  Custom CNN

  - Framework: TensorFlow / Keras
  - Architecture: Sequential CNN with 3 convolutional blocks (32, 64, 128
  filters), batch normalization, dropout, and global average pooling
  - Parameters: 110,785
  - Input: 150x150 RGB images
  - Output: Binary fire probability (sigmoid)
  - Best Validation Accuracy: ~89.5%

  YOLOv8 Nano

  - Framework: Ultralytics YOLOv8
  - Base Model: YOLOv8n (nano), fine-tuned for fire detection
  - Input: 150x150 images
  - Output: Object detection bounding boxes with fire/non-fire classification

  Dataset

  - Source:
  https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data
  (Kaggle)
  - Total Images: ~3,300
  - Split: 80% training (2,638) / 20% validation (662)
  - Classes: Fire, Non-fire (binary)
  - Augmentation: Shear, zoom, horizontal flip, rescaling to [0, 1]

  Features

  - Model Training & Evaluation — Train and compare both CNN and YOLO models on
  the same dataset
  - Video Processing — Frame-by-frame fire detection on real video footage
  - Grad-CAM Visualization — Interpretability heatmaps showing what the CNN
  focuses on when detecting fire
  - Model Comparison — Side-by-side frame-level detection plots comparing both
  architectures
  - Annotated Output Videos — Generated MP4s with detection overlays and
  confidence scores

  Requirements

  - Python 3.11+
  - TensorFlow / Keras
  - Ultralytics (YOLOv8)
  - OpenCV (cv2)
  - NumPy
  - Matplotlib
  - Kaggle Hub

  Getting Started

  1. Open AIFireAndSmokeDetectionResearch.ipynb in Google Colab or Jupyter
  Notebook.
  2. Run all cells to download the dataset, train both models, and evaluate on
  test videos.
  3. Pre-trained model weights are included:
    - fire_detection_cnn.keras — Custom CNN
    - best.pt — YOLOv8 Nano

  Project Structure

  ├── AIFireAndSmokeDetectionResearch.ipynb   # Main research notebook
  ├── fire_detection_cnn.keras                # Trained CNN model weights
  ├── best.pt                                 # Trained YOLOv8 model weights
  ├── videoplayback.mp4                       # Test video
  ├── Yosemite Forest Fire Time Lapse and Flyover.mp4  # Test video
  └── README.md

  Results

  The custom CNN achieves up to ~89.5% validation accuracy on fire/non-fire
  classification. Both models are evaluated on real wildfire footage, with
  Grad-CAM heatmaps providing visual interpretability for the CNN's predictions.
   Frame-level comparison plots demonstrate the detection behavior of each model
   over time.
