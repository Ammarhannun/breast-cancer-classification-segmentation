# Breast Cancer Classification Using Deep Learning on Ultrasound Images

## Project Overview
This project applies deep learning to classify breast ultrasound images into three categories: benign, malignant, and normal.  
By using transfer learning with ResNet50, the model leverages pre-trained ImageNet weights to achieve strong performance while reducing training time and computational cost.  

---

## Objectives
- Develop a robust classification pipeline for medical imaging.  
- Apply transfer learning with ResNet50 to efficiently extract and fine-tune features.  
- Evaluate performance using multiple metrics including accuracy, precision, recall, and F1-score.  
- Demonstrate the potential of AI for early breast cancer detection.  

---

## Dataset
- **Source:** Breast Ultrasound Image Dataset (publicly available).  
- **Classes:** Benign, Malignant, Normal.  
- **Data Preparation:** Images were resized to 224×224 pixels, normalized, and augmented through rotation, flipping, and contrast adjustments.  
- **Splits:** Training, validation, and test sets were created to ensure reliable evaluation.  

---

## Methodology
1. **Preprocessing:** Normalization, augmentation, and dataset splitting.  
2. **Model Architecture:**  
   - ResNet50 pre-trained on ImageNet.  
   - Final layers adapted for three-class classification.  
   - Dropout and dense layers added for regularization.  
3. **Training:**  
   - Optimizer: Adam.  
   - Loss function: Categorical cross-entropy.  
   - Early stopping and learning rate scheduling applied.  
4. **Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrix analysis.  

---

## Results
- The model achieved strong classification accuracy across all three classes.  
- Performance was balanced between benign, malignant, and normal images.  
- Training and validation curves, as well as confusion matrices, were used to assess effectiveness.

---

## Future Work
- Extend to segmentation tasks using architectures such as U-Net.  
- Experiment with alternative classification models such as DenseNet or EfficientNet.  
- Deploy as a prototype diagnostic application using Flask or Streamlit.  

---

## References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.  
- Dataset: Breast Ultrasound Images Dataset (Kaggle).  
