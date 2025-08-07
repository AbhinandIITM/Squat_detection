# Squat Detection and Correction using MediaPipe and GNN

This project provides a real-time squat detection and correction system using computer vision and deep learning. It leverages the MediaPipe library for accurate pose estimation and a Graph Neural Network (GNN) to classify squat form and provide instant feedback.

## Features

- **Real-time Pose Estimation**: Utilizes MediaPipe to detect 33 key body landmarks from a video feed (webcam or file).
- **Squat Counting**: Accurately counts the number of squats performed.
- **Form Analysis**: Calculates critical joint angles (hip, knee, ankle) to analyze the user's squat form.
- **Corrective Feedback**: Employs a trained GNN model to classify each squat as correct or incorrect and provides real-time textual feedback to help the user improve their posture (e.g., "Lower your hips," "Bend your knees more").
- **Data-Driven Approach**: Includes scripts to process videos, extract pose data, and generate a labeled dataset (`squat_dataset.csv`) for training the classification model.

## How It Works

The system follows a multi-stage process to deliver real-time squat analysis:

1.  **Video Input**: The application captures video from a webcam or a pre-recorded file.
2.  **Pose Estimation**: For each frame, MediaPipe's Pose solution is used to identify the coordinates of 33 body landmarks.
3.  **Feature Extraction**: Key landmarks (shoulders, hips, knees, ankles) are isolated. The system calculates the angles between these joints to quantify the squat's form. The landmark coordinates are normalized relative to the hip's center to ensure the model is robust to changes in camera distance and user position.
4.  **GNN Classification**: The extracted landmark data is structured as a graph. A pre-trained Graph Neural Network (GNN) model processes this graph to classify the squat's correctness.
5.  **Feedback and Display**: The system overlays the joint angles, squat count, and corrective feedback directly onto the video feed, providing an interactive and intuitive experience for the user.

## Technology Stack

- **Python**: The core programming language.
- **OpenCV**: Used for video capture, processing, and displaying the output.
- **MediaPipe**: For high-fidelity body pose tracking.
- **PyTorch**: The primary deep learning framework.
- **PyTorch Geometric (PyG)**: An extension library for PyTorch for implementing Graph Neural Networks.
- **NumPy**: For efficient numerical operations.
- **Pandas**: Used for data manipulation and creating the training dataset.
- **Jupyter Notebook**: For code development, experimentation, and visualization.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AbhinandIITM/Squat_detection.git
    cd Squat_detection
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file is not provided, but you can install the necessary packages using pip:
    ```bash
    pip install opencv-python mediapipe torch torchvision torchaudio torch-geometric pandas jupyter ultralytics
    ```

## Usage

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open and Run `squat.ipynb`:**
    - Open the `squat.ipynb` file in the Jupyter interface.
    - The notebook contains all the code for data processing, model training, and real-time squat detection.
    - You can modify the `cv2.VideoCapture()` input to use your webcam (e.g., `cv2.VideoCapture(0)`) or a different video file.

## File Descriptions

- **`squat.ipynb`**: The main Jupyter Notebook containing the complete workflow.
- **`squat_dataset.csv`**: The dataset generated and used for training the GNN model. Each row represents a single frame's pose data and a corresponding label.
- **`squat_video.mp4`**, **`squat05.mp4`**: Sample video files used for demonstration and data collection.
- **`README.md`**: This file.

## Future Improvements

- **Enhance the Dataset**: Train the model on a larger and more diverse dataset of squat variations to improve accuracy and generalization.
- **Improve the GNN Model**: Experiment with different GNN architectures and hyperparameters to achieve better performance.
- **Web Application**: Deploy the system as a web application using a framework like Flask or FastAPI for easier access.
- **Expand Exercise Library**: Extend the system to recognize and provide feedback for other exercises like lunges, push-ups, or deadlifts.
