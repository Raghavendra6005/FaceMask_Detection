# Face Mask Detection using SVM

![Face Mask Detection](images/face_mask_detection.png)

This project demonstrates a simple face mask detection system using an SVM (Support Vector Machine) classifier. The SVM model is trained to detect whether a person is wearing a mask or not. The project utilizes OpenCV for image processing and scikit-learn for building the SVM model.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
- [SVM Model](#svm-model)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## About

Face mask detection has become a crucial technology in the context of public health and safety. This project provides a basic demonstration of face mask detection using a simple SVM model. The SVM classifier is trained on a dataset of images featuring individuals with and without masks.

## Getting Started

### Prerequisites

Before you start, make sure you have the following prerequisites:

- Python 3
- OpenCV (`pip install opencv-python`)
- scikit-learn (`pip install scikit-learn`)

## Usage

1. Clone the repository to your local machine.
2. Prepare your dataset by organizing images of people with and without masks in the `dataset` folder.
3. Adjust the SVM model and parameters in `svm_mask_detection.py` as needed.
4. Run `svm_mask_detection.py` using a Python interpreter to start face mask detection.

## SVM Model

The SVM model (in `svm_model.py`) is trained to classify whether a person is wearing a mask or not. You can generate this model after training your dataset.

## Project Structure

The project structure is organized as follows:

- `main.py`: The main application code.
- `svm_mask_detection.py`: The SVM model code for face mask detection.
- `svm_model.py`: The trained SVM model (to be generated after training).
- `dataset/`: Organize your dataset of images here.
- `images/`: Store images and GIFs for the README.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests to help improve this project.

## License

This project is licensed under the [MIT License](LICENSE).

![Face Mask Detection GIF](images/face_mask_detection.gif)
