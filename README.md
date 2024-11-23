# FACE DETECTION AND RECOGNITION

# Face Detection and Recognition Project

This project demonstrates a face detection and recognition system using OpenCV and Haar Cascade classifiers. The project involves two main scripts: `create_data.py` for collecting facial data and `face_recognise.py` for recognizing faces in real-time. A pre-trained Haar Cascade XML file is used for face detection.

## Project Overview

### 1. Data Collection (`create_data.py`)
The `create_data.py` script captures images from a webcam to build a dataset of facial images. The images are grayscale and resized to a standard dimension.

### 2. Face Recognition (`face_recognise.py`)
The `face_recognise.py` script trains a Local Binary Patterns Histograms (LBPH) face recognizer using the collected data and recognizes faces in a live video stream.

## Prerequisites

1. Python 3.x
2. OpenCV library

To install OpenCV, use:
```bash
pip install opencv-python opencv-contrib-python
```

## Setup and Usage

### Step 1: Clone the Repository
Clone this repository and navigate to the project folder.

### Step 2: Dataset Preparation

1. Create a folder named `datasets` in the project directory.
2. Inside `datasets`, create a subfolder for each person's images (e.g., `datasets/<person_name>`).
3. Run the `create_data.py` script to populate the dataset:
   ```bash
   python create_data.py
   ```

   - The script captures images from the webcam.
   - A total of 23 images are saved for the specified individual in the `datasets/<person_name>` folder.

### Step 3: Face Recognition

1. Ensure the `datasets` folder contains at least one person's image dataset.
2. Run the `face_recognise.py` script to initiate the recognition system:
   ```bash
   python face_recognise.py
   ```

   - The script trains the LBPH recognizer with the images in `datasets`.
   - It uses the webcam feed to detect and recognize faces.

### Step 4: Haar Cascade Classifier

The `haarcascade_frontalface_default.xml` file is a pre-trained model provided by OpenCV for detecting faces. It identifies regions in an image that may contain a face. This file is utilized in both `create_data.py` and `face_recognise.py` scripts through OpenCV's `CascadeClassifier` class.

## Code Details

### `create_data.py`

- **Purpose**: Captures images of a person's face to build a dataset.
- **Key Functions**:
  - `cv2.VideoCapture(0)`: Initializes webcam.
  - `cv2.CascadeClassifier(haar_file)`: Loads the Haar Cascade for face detection.
  - `cv2.imwrite()`: Saves the detected face images.
- **Loop**: Captures 23 images, converts them to grayscale, and detects faces using `detectMultiScale`.
- **Output**: Images saved in `datasets/<sub_data>`.

### `face_recognise.py`

- **Purpose**: Recognizes faces in real-time using the LBPH face recognizer.
- **Steps**:
  1. Loads the dataset images and labels.
  2. Trains the LBPH face recognizer (`cv2.face.LBPHFaceRecognizer_create()`).
  3. Detects faces in the webcam feed using the Haar Cascade.
  4. Predicts the identity of the detected face.
- **Output**: Displays the webcam feed with the detected face's name and confidence score.

### Haar Cascade XML

The `haarcascade_frontalface_default.xml` file contains pre-trained data for face detection using Haar features. This file is essential for both scripts to detect faces accurately.

## File Structure

```
project/
|-- haarcascade_frontalface_default.xml
|-- create_data.py
|-- face_recognise.py
|-- datasets/
    |-- <person_name>/
        |-- 1.png
        |-- 2.png
        ...
```

## Notes

1. Press `ESC` to exit the scripts.
2. Ensure adequate lighting for better face detection and recognition.
3. For multiple users, create separate subfolders under `datasets` and run `create_data.py` for each user.

## Known Issues and Limitations

- Recognition may fail under poor lighting conditions or significant changes in facial features.
- Limited to frontal face detection; may not work well with side profiles.

## Future Enhancements

- Improve detection using deep learning models like DNNs.
- Add support for recognizing multiple faces simultaneously.

## License

This project is licensed under the MIT License.

