# Attention-Enhanced Pothole Detection with Sensor Fusion

This project implements a deep learning model to detect potholes from road images. It uses a ResNet-18 backbone enhanced with Channel and Spatial Attention mechanisms (CBAM). The system is presented through a Gradio web interface that allows for model training, evaluation, and inference with simulated multi-sensor fusion.

![Gradio App Screenshot](https://user-images.githubusercontent.com/your-username/your-repo/assets/screenshot.png) <!-- Replace with a link to a screenshot of your app -->

## Features

- **Attention-Enhanced Model**: Uses a CBAM-like attention block on a ResNet-18 backbone for improved feature extraction.
- **Data Augmentation**: Employs a comprehensive set of augmentations to create a robust model.
- **Sensor Fusion Simulation**: A Gradio interface to simulate and fuse readings from a camera (CNN), LiDAR, Ultrasonic, and GPR.
- **Dynamic Weighting**: Features an "Ambiguous Mode" that dynamically adjusts sensor weights based on their confidence.
- **Full Training Pipeline**: Train the model on a custom dataset and evaluate it with standard metrics (Accuracy, Precision, Recall, F1, AUC).
- **Rich Visualizations**: Generates plots for training history, confusion matrix, and ROC curve.

## Project Structure

```
.
├── pothole_detection_12_Apr_working.py  # Main application script
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Files to be ignored by Git
└── README.md                            # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The PyTorch installation can be hardware-specific. Please visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command for your system. For a typical CUDA setup, you might run:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    Then, install the remaining packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    ```bash
    python pothole_detection_12_Apr_working.py
    ```
    This will launch a Gradio web server. Open the provided URL (e.g., `http://127.0.0.1:7860`) in your browser.

2.  **Train the Model (Tab 1):**
    -   Prepare your dataset as a `.zip` file. It must contain two subfolders: `Normal` and `Pothole`.
    -   Upload the `.zip` file and click "Train Model".
    -   Once training is complete, evaluation metrics and plots will be displayed. The best model is saved as `best_pothole_model.pth`.

3.  **Inference & Sensor Fusion (Tab 2):**
    -   Upload a test image.
    -   To simulate a challenging scenario, check the "Ambiguous Mode" box. This will allow you to adjust sliders for other simulated sensors.
    -   Click "Analyze" to see the model's prediction, the fused decision, and a breakdown of sensor contributions.