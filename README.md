# COVID-19 Classification from CT-Scans using ResNet-50

## 1. Project Overview
This project uses Deep Learning (CNN) to classify COVID-19 and Non-COVID cases from CT-scan images. It utilizes the **ResNet-50** architecture with a **Transfer Learning** methodology to achieve high precision in medical image classification.

## 2. Dataset Specifications
* **Source:** SARS-CoV-2 CT-Scan Dataset (Kaggle).
* **Composition:** ~3,700 high-resolution CT-scan images.
* **Classes:** 1. `COVID` (Infected cases)
  2. `non-COVID` (Healthy/Normal or other lung infections)
* **Pre-processing:** All images were resized to 224x224, normalized using ImageNet stats, and augmented with Random Horizontal Flips.

## 3. Technical Methodology
### Model Architecture
* **Base Model:** ResNet-50 (Pre-trained weights).
* **Modification:** Early layers were frozen; the final `fc` layer was replaced with a custom Linear layer for binary classification.
* **Fine-tuning:** Only `layer4` and the classifier were unfrozen for domain-specific weight updates.

### Hyperparameters & Strategy
* **Optimizer:** AdamW (Learning Rate: 1e-4).
* **Loss Function:** CrossEntropyLoss.
* **Data Split:** 70% Train | 15% Validation | 15% Test (Stratified Split).
* **Batch Size:** 32.
* **Training Epochs:** 12.

## 4. Performance Metrics (Final Results)
Final evaluation results on the unseen **Test Set** (15% of the data):

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **98.00%** |
| **AUC-ROC Score** | **0.9961** |
| **Validation Accuracy** | **98.57%** |
| **Training Accuracy** | **98.92%** |

### Loss & Accuracy Curves
![Loss & Accuracy Curves](./Assets/abc.jpeg) 
*(Note: Validation loss closely tracked training loss, indicating zero overfitting.)*

### Confusion Matrix
![Confusion Matrix](./Assets/def.jpeg)
* The model correctly identified 367 out of 373 test samples, with near-perfect sensitivity for Non-COVID cases.

## 5. Repository Structure
* `ResNet-50.ipynb`: Complete training and evaluation pipeline.
* `resnet50_covid_best.pth`: Saved model weights for deployment.

## 6. How to Reproduce
1. Install dependencies: `pip install torch torchvision scikit-learn seaborn matplotlib`
2. Place your `COVID` and `non-COVID` image folders in the root directory.
3. Run the notebook `ResNet-50.ipynb` to re-train or evaluate using the `.pth` file.

---
**Author:** Dhruv Rai (23bcs028) , Diwakar Nagar (23bcs029)
**Institute:** Shri Mata Vaishno Devi University (SMVDU)