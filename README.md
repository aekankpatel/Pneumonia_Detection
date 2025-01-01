# Pneumonia Detection Using CNN and MobileNet

This project implements a robust approach to pneumonia detection using deep learning models. It combines a custom Convolutional Neural Network (CNN) and a pre-trained MobileNet model to achieve high accuracy through an ensemble method. The project uses chest X-ray images classified as either "Normal" or "Pneumonia."

---

## Features
- Custom CNN for feature extraction and classification.
- Transfer learning with MobileNet, pre-trained on ImageNet.
- Ensemble of CNN and MobileNet to enhance prediction accuracy.
- Visualizations for training performance and prediction distributions.

---

## Project Workflow

### 1. Data Preparation

#### Dataset Structure
The dataset is organized into directories:
- **Training Data:**
  - `Train_Normal`
  - `Train_Pneumonia`
- **Testing Data:**
  - `Test_Normal`
  - `Test_Pneumonia`

#### Data Augmentation
- **Image Rescaling:** Pixel values are normalized to the range [0, 1].
- **Validation Split:** 15% of the training data is reserved for validation.

#### Data Summary
- Training Set: 11,456 images
- Validation Set: 2,021 images
- Testing Set: 2,379 images

### 2. CNN Model

#### Architecture
- **Input Layer:** Image dimensions (224x224x3)
- **Convolutional Layers:** 3 layers with ReLU activation and MaxPooling.
- **Fully Connected Layer:**
  - Dense layer with 128 units and Dropout (0.5).
  - Output layer with 1 unit and sigmoid activation for binary classification.

#### Training
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Stopping Criteria:** Early stopping after 2 epochs without improvement in validation loss.

#### Results
- Final Validation Accuracy: ~98%

### 3. MobileNet Model

#### Architecture
- **Base Model:** Pre-trained MobileNet (ImageNet weights).
- **Custom Layers:**
  - Global Average Pooling
  - Dropout (0.5)
  - Dense layer with sigmoid activation.
- **Freezing:** All layers in the MobileNet base are frozen.

#### Training
- Same configuration as the CNN model.
- Trained for 10 epochs.

#### Results
- Final Validation Accuracy: ~96%

### 4. Ensemble Model

#### Methodology
- Predictions from the CNN and MobileNet models are averaged to form the ensemble.
- Final predictions are thresholded at 0.5 for binary classification.

#### Results
- **Accuracy:** 97.2%
- **Precision:** 97.5%
- **Recall:** 97.7%
- **F1 Score:** 97.6%

---

## Visualization

1. **Training Performance:**
   - Plots for accuracy and loss trends during training for CNN and MobileNet.

2. **Probability Histograms:**
   - Visualizations of predicted class probabilities for CNN, MobileNet, and the ensemble.

3. **Confusion Matrices:**
   - Confusion matrices for each model and the ensemble, highlighting their classification performance.

---

## Installation

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install tensorflow numpy scikit-learn matplotlib
  ```

### Dataset
Ensure the dataset is structured as described under **Data Preparation.**

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pneumonia-detection
   ```
2. Navigate to the project directory:
   ```bash
   cd pneumonia-detection
   ```
3. Run the script:
   ```bash
   python main.py
   ```

---

## Results

| Model       | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|-----------|--------|----------|
| CNN         | 96.6%    | 97.1%     | 97.0%  | 97.1%    |
| MobileNet   | 95.0%    | 95.6%     | 95.8%  | 95.7%    |
| Ensemble    | **97.2%**| **97.5%** | **97.7%**| **97.6%**|

---

## Future Improvements
- Fine-tune the MobileNet model by unfreezing selective layers.
- Explore additional ensemble methods like weighted averaging.
- Evaluate model performance on external datasets.

---
