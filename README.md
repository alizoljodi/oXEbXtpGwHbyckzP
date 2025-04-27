# Page Flip Detection Project

## Overview
This project aims to classify whether an image, captured from a video feed, represents a **page flipping** action or **not flipping**. It explores various computer vision and machine learning techniques, ranging from traditional feature engineering to deep learning models.

## Project Structure
- **Conventional Machine Learning Models:**
  - Feature extraction using HOG, LBP, Raw Pixels, and Thresholded images.
  - Models: SVM, Random Forest, Logistic Regression, k-Nearest Neighbors.
- **Deep Learning Models:**
  - Fine-tuned **Pretrained ResNet18** from TorchVision.
  - Custom-built **LeNet** architecture.
- **Evaluation Metrics:**
  - F1-Score is the primary metric for model evaluation.

## Dataset
- Two main splits: **training** and **testing**.
- Each split contains two subfolders: **flip** and **notflip**.
- Images are grayscale frames extracted from smartphone video recordings.

## Results Summary
- **Pretrained ResNet18** achieved the highest F1 scores.
- **Conventional ML models + HOG Features** provided competitive results.
- **Raw Pixel** and **Threshold Features** combined with classical models also performed very well.
- **LeNet** also reached above 95% F1-score but slightly lower than ResNet18 and best traditional models.

## Key Takeaways
- Simpler models performed very well due to the **small dataset size** and **simple classification challenge**.
- Heavy CNN models beyond ResNet18 are **not recommended** as they could cause overfitting without significant performance gain.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/page-flip-detection.git
   cd page-flip-detection
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset inside the `data/` directory following the mentioned structure.

4. Run the main notebook or Python scripts:
   ```bash
   python train_and_evaluate.py
   ```

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- scikit-learn
- matplotlib
- torchvision

### Author
**Ali Zoljodi**
