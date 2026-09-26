# Drone Gesture Classification

Drone Gesture Classification is a computer-vision prototype that collects webcam images of hand gestures, trains a TensorFlow/Keras convolutional neural network to classify nine drone commands, and displays the predicted command from a live camera feed. A separate MediaPipe-based script provides landmark-driven gesture recognition without using the CNN model.

## Architecture and overview

The repository is organized as a small, script-oriented Python application:

```text
Webcam
  ├─ data.py ───────────────► dataset/<gesture>/*.jpg
  ├─ trainer2.py ───────────► drone_cnn.keras
  ├─ test.py ───────────────► live CNN predictions and command overlay
  ├─ check.py ──────────────► classification report and confusion matrix
  └─ test_mediapipe.py ─────► landmark-based gesture commands
```

The generated architecture diagram is available at [GitDiagram](https://gitdiagram.com/Ritwik389/drone-gesture-classfication). The CNN path uses OpenCV for camera input, Keras image generators for directory-based data loading, and a saved Keras model for inference. The MediaPipe path processes hand landmarks directly and is independent of the CNN training pipeline.

## Key features

- Webcam dataset collection for nine gesture classes: `rest`, `take_off`, `land`, `up`, `down`, `left`, `right`, `flip`, and `capture`.
- CNN training with augmentation, validation splitting, learning-rate reduction, and early stopping.
- Saved trained model in `DRONE PROJECT CNN/drone_cnn.keras`.
- Live webcam inference with confidence thresholding and per-class probabilities.
- Model evaluation with a classification report and confusion matrix.
- Alternative MediaPipe hand-landmark gesture recognition.

## Tech stack

- Python
- OpenCV (`cv2`) for webcam capture and image processing
- TensorFlow/Keras for model training and inference
- NumPy for array processing
- scikit-learn for evaluation metrics
- Matplotlib and Seaborn for evaluation plots
- MediaPipe for landmark-based gesture recognition

The repository currently has no `requirements.txt`, `pyproject.toml`, Conda environment file, Dockerfile, or other dependency lockfile.

## Setup and installation

1. Install Python and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   ```

2. Install the dependencies used by the scripts:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install tensorflow opencv-python numpy scikit-learn matplotlib seaborn mediapipe
   ```

3. Change into the script directory:

   ```bash
   cd "DRONE PROJECT CNN"
   ```

   The camera scripts require a working webcam and permission to access it. The training and evaluation scripts require a populated `dataset/` directory arranged in one subdirectory per gesture class. The dataset is intentionally ignored by Git.

## Usage

Collect training images by pressing the number keys `0` through `8` while `data.py` is running; press `q` to stop:

```bash
python data.py
```

Train and save the CNN:

```bash
python trainer2.py
```

Run live CNN classification:

```bash
python test.py
```

Generate evaluation metrics and a confusion matrix:

```bash
python check.py
```

Run the independent MediaPipe landmark recognizer:

```bash
cd ..
python test_mediapipe.py
```

## Status

**Incomplete — research/prototype stage.**

The core collection, training, evaluation, CNN inference, and MediaPipe scripts are present, and a trained `drone_cnn.keras` artifact is checked in. The following concrete gaps remain:

- There is no dependency manifest or reproducible environment configuration.
- There are no automated unit, integration, or model-quality tests; the only test-named files are interactive webcam programs.
- There is no CI/build configuration.
- The dataset required for retraining is absent from the repository because `DRONE PROJECT CNN/.gitignore` excludes `dataset/`; users must collect it locally.
- `DRONE PROJECT CNN/test.py` uses a hard-coded region beginning at x-coordinate `1200`, so cameras with smaller frames can produce an empty/invalid ROI and fail during resizing.
- The scripts display recognized commands but do not connect them to a drone SDK or send flight-control commands.
- The training script assumes all nine class directories exist and does not validate dataset completeness before training.
- Camera-read failures terminate the interactive loops without reporting a diagnostic.

These issues should be addressed before treating the project as a reliable or deployable drone-control system.

## License

No license file is present in the repository.
