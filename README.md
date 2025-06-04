# Brain Tumor Classification using EfficientNet and GradCAM

A deep learning-based system for automated classification of brain tumors from MRI images with visual explanations.

## Overview

This project implements a brain tumor classification system using EfficientNetB1 architecture with transfer learning and GradCAM visualization. The model can classify brain MRI images into four categories:

- **Glioma**
- **Meningioma** 
- **Pituitary Tumor**
- **No Tumor**

## Key Features

- **High Accuracy**: Achieves 98.55% classification accuracy
- **Visual Explanations**: Integrated GradCAM for model interpretability
- **Efficient Architecture**: Uses EfficientNetB1 for balanced performance and computational efficiency
- **Data Augmentation**: Comprehensive preprocessing pipeline to handle limited medical data
- **Transfer Learning**: Pre-trained on ImageNet and fine-tuned for brain tumor classification

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.55% |
| **Precision** | 98% |
| **Recall** | 98% |
| **F1-Score** | 98% |

### Class-wise Performance
- Glioma: 99% precision, 96% recall
- Meningioma: 96% precision, 98% recall
- No Tumor: 99% precision, 100% recall
- Pituitary: 99% precision, 99% recall

## Requirements

```
tensorflow>=2.4.1
keras>=2.4.0
opencv-python>=4.5.1
numpy>=1.19.5
pandas>=1.2.3
matplotlib>=3.3.4
scikit-learn>=0.24.1
```



### Training the Model

```python
python train.py --data_path ./dataset --epochs 30 --batch_size 32
```

### Making Predictions

```python
python predict.py --model_path ./models/best_model.h5 --image_path ./test_image.jpg
```

### Generating GradCAM Visualizations

```python
python gradcam_viz.py --model_path ./models/best_model.h5 --image_path ./test_image.jpg
```

## Architecture

- **Base Model**: EfficientNetB1 (pre-trained on ImageNet)
- **Custom Head**: Global Max Pooling → Dropout (0.5) → Dense (4 classes)
- **Input Size**: 240×240×3
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Cross-entropy

## Data Preprocessing

- Image resizing to 240×240 pixels
- Normalization to [0,1] range
- Data augmentation:
  - Rotation (±30°)
  - Zoom (±20%)
  - Horizontal flipping
  - Width/height shifts (±10%)

## Results

The model outperforms existing state-of-the-art methods while maintaining computational efficiency. GradCAM visualizations provide interpretable heatmaps showing which brain regions influence the classification decisions.


## Acknowledgments

- EfficientNet architecture by Google Research
- GradCAM implementation for model interpretability
- Medical imaging datasets used for training and evaluation
