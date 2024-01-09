# DeepBoneFractureDetection

## Overview
This repository presents the implementation and findings of a study on the efficacy of deep learning models in bone fracture detection using X-ray images. The methodology encompasses data preparation, model selection, training, and evaluation processes.

## Methodology

### Dataset
- **Source:** Kaggle repository ([Bone Fracture Detection using X-rays](https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays))
- **Type:** X-ray images
- **Classes:** "fracture" and "not fracture"
- **Distribution:**
  - Training set: 8,863 images
  - Validation set: 600 images

### Setting up the Environment for the Code
- **Programming language:** Python is chosen for its extensive libraries and frameworks for deep learning.
- **Deep learning framework:** TensorFlow or PyTorch is used to build and train the deep learning models.
- **Libraries:**
  - NumPy: For numerical computations and array manipulation.
  - Pandas: For data manipulation and analysis.
  - Matplotlib and OpenCV: For image visualization and processing.
  - Scikit-learn: For performance evaluation and metrics.

### Preprocessing and Data Generation
- **Image resizing:** All X-ray images are resized to a consistent dimension (e.g., 224x224 pixels) for compatibility with deep learning models.
- **Normalization:** Pixel values are normalized to a range of 0 to 1 for better model convergence.
- **Data augmentation:** To artificially increase the training dataset size and diversity, the following techniques are applied:
  - Random rotations: Images are randomly rotated within a specified range.
  - Random flipping: Images are flipped horizontally or vertically.
  - Random cropping: Random sections of images are cropped.
  - Gaussian noise addition: Random noise is added to images to simulate real-world variations.

### Models
- **Pre-trained models:** The study explores the performance of several pre-trained deep learning models, including:
  - VGG16
  - ResNet50
  - DenseNet121
  - EfficientNetB0
  - EfficientNetB3
- **Custom CNN object detection model:** A custom convolutional neural network (CNN) model is designed specifically for fracture localization.
- **Fine-tuning:** Pre-trained models are fine-tuned, adjusting their final layers to adapt to the fracture detection task.

## How to Use
1. Clone the repository: `git clone https://github.com/your-username/DeepBoneFractureDetection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the instructions in the `src` directory to train and evaluate the models.

## Repository Structure
- `src/`: Contains the source code for model training, evaluation, and analysis.
- `data/`: Placeholder for the dataset; download and organize the data before running the code.
- `results/`: Stores the results of model evaluations and analysis.

## Contributors
- Muhammad Musa

Feel free to contribute, report issues, and provide feedback!

**Index Terms:** X-Ray, Image Detection, CNN, VGG16, ResNet50, DenseNet121, EfficientNetB0, EfficientNetB3.
