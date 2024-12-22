# Sleep Stage Classification with AI

This repository contains a machine learning project aimed at classifying sleep stages based on EEG signal data. The dataset provided consists of 30-second EEG signal spectrograms, preprocessed into 51×59 matrices. The classification challenge is to identify five sleep stages: Wake (0), NREM1 (1), NREM2 (2), NREM3 (3), and REM (4).

## Problem Statement

Sleep is a critical physiological process, and understanding its stages has significant implications for neuroscience, health, and medical diagnostics. Accurate classification of sleep stages from EEG signals can aid in diagnosing sleep disorders and improving health monitoring technologies.

The dataset for this project poses several challenges:

- **Class Imbalance**: The dataset distribution is highly skewed, making it difficult to train a model that performs well across all classes.
- **Time-Series Nature**: EEG signals are inherently temporal, but the dataset consists of spectrogram-based snapshots, potentially losing some temporal context.
- **Dataset Complexity**: Each sample is represented as a 51×59 spectrogram matrix, making preprocessing and model training computationally intensive.

## Dataset Overview

The dataset is organized into five classes corresponding to the sleep stages:

| Class Label | Sleep Stage          | Sample Count |
| ----------- | -------------------- | ------------ |
| 0           | Wake                 | 2906         |
| 1           | NREM1 (Light Sleep)  | 237          |
| 2           | NREM2 (Deeper Sleep) | 1755         |
| 3           | NREM3 (Deep Sleep)   | 497          |
| 4           | REM (Dream Sleep)    | 418          |

### Key Observations:

- The **Wake** stage dominates the dataset, constituting \~44% of the samples.
- The **NREM1** class is significantly underrepresented, making it the most challenging stage to classify.

### Dataset Source

[Download the dataset here](https://drive.google.com/file/d/1Y2cTYR_t_10NAbznspE5bBjuATPdTgtq)

## Methodology

### Data Preprocessing

1. **Loading and Encoding**:

   - `.npy` files containing spectrogram data were loaded.
   - Labels were encoded using `LabelEncoder`.

2. **Balancing the Dataset**:

   - Oversampling/undersampling was used to balance the classes, targeting 1000 samples per class.
   - Resampled data resulted in a balanced dataset for training.

3. **Normalization**:

   - Data was normalized using `MinMaxScaler` to ensure uniform feature scaling.

### Model Architecture

A Convolutional Neural Network (CNN) was designed to classify sleep stages. The architecture includes:

- **Input Layer**: Accepting 51×59×1 matrices.
- **Convolutional Layers**: Extracting spatial features using a series of Conv2D layers.
- **MaxPooling Layers**: Reducing dimensionality while preserving important features.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout**: Preventing overfitting.

#### Model Summary

```plaintext
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 49, 57, 32)       320       
conv2d_2 (Conv2D)            (None, 49, 57, 32)       9248      
max_pooling2d_1 (MaxPooling2D) (None, 24, 28, 32)     0         
conv2d_3 (Conv2D)            (None, 24, 28, 32)       9248      
conv2d_4 (Conv2D)            (None, 24, 28, 32)       9248      
max_pooling2d_2 (MaxPooling2D) (None, 12, 14, 32)     0         
conv2d_5 (Conv2D)            (None, 12, 14, 64)       18496     
conv2d_6 (Conv2D)            (None, 12, 14, 64)       36928     
max_pooling2d_3 (MaxPooling2D) (None, 6, 7, 64)       0         
conv2d_7 (Conv2D)            (None, 6, 7, 128)        73856     
max_pooling2d_4 (MaxPooling2D) (None, 3, 3, 128)      0         
flatten_1 (Flatten)          (None, 1152)             0         
dropout_1 (Dropout)          (None, 1152)             0         
dense_1 (Dense)              (None, 512)              590336    
dropout_2 (Dropout)          (None, 512)              0         
dense_2 (Dense)              (None, 5)                2565      
=================================================================
Total params: 740,245
Trainable params: 740,245
Non-trainable params: 0
_________________________________________________________________
```

### Training

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 16
- **Epochs**: 25

### Results

#### Accuracy and Loss Curves

Plots of training and validation loss and accuracy show the model’s performance over epochs:

- **Accuracy**: The training accuracy reached \~85.8%, while the validation accuracy plateaued at \~67.3%.
- **Loss**: The loss curve indicated slight overfitting beyond epoch 18.

![image](https://github.com/user-attachments/assets/cccc3ae4-67e1-4bff-888e-835265e9e3f6)

![image](https://github.com/user-attachments/assets/762b806c-8f57-4d93-95ab-5e18fd02284e)


#### F1 Score

The micro F1 score on the test set was **0.66**.

### Key Challenges

- **Imbalanced Dataset**: Despite balancing, the model struggled with minority classes like NREM1 and REM.
- **High Variance**: Overfitting was observed, necessitating further regularization or data augmentation.

## Conclusion

This project demonstrates the feasibility of classifying sleep stages from EEG spectrogram data using deep learning. Future work could involve:

1. Leveraging temporal dependencies using Recurrent Neural Networks (RNNs) or Transformer-based models.
2. Augmenting the dataset with synthetic samples to address class imbalance.
3. Exploring ensemble methods to improve robustness.

## How to Run

1. Download and extract the dataset into the `data/` directory.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python Dreamy.py
   ```

---

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.



