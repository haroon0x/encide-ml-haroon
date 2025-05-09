
# Image Classifier - encide-ml
This repository contains code and resources for training and evaluating image classification models to distinguish between cats and dogs using TensorFlow/Keras.  
As part of the **encide-ml competition**.

## 📚 Table of Contents
- [Project Overview](#project-overview)
- [Training Summary](#-training-summary)
- [Technologies & Dependencies](#technologies--dependencies)
- [Workflow](#workflow)
- [Dataset Structure](#dataset-structure)
- [How to Run](#how-to-run)
- [Acknowledgements](#acknowledgements)
---

Deployed on Gradio : https://dd385c080c430f1f47.gradio.live
## Project Overview

- **Custom Convolutional Neural Network (CNN):**  
    Built from scratch with Conv2D, BatchNormalization, Dropout, and MaxPooling layers. Uses GlobalAveragePooling2D and dense layers for classification. Data augmentation and normalization are applied for better generalization.

- **Transfer Learning with ResNet-50:**  
    Uses a pretrained ResNet-50 model (`include_top=False`) as a feature extractor, with custom dense layers on top for binary classification. The base model is initially frozen, then partially unfrozen for fine-tuning. Preprocessing includes normalization and ResNet-50-specific input scaling. Data augmentation is applied.
- Training includes early stopping, learning rate scheduling, and model checkpointing to prevent overfitting and ensure the best results.
- models are exported in Keras (`.keras`) and TensorFlow Lite (`.tflite`) formats for deployment or further use.

> **Note:**  
> For this submission, I have trained the ResNet-50 model and am submitting this model as my final solution.  
> The first model I trained achieved a validation accuracy of **76.4%** (AUC: 0.8508, Precision: 0.7225, Recall: 0.8692, Loss: 0.5540).  
> The new ResNet-50 model achieves a **much higher validation accuracy of 98.8%** (AUC: 0.9993, Precision: 0.9868, Recall: 0.9890, Loss: 0.0297),  
> showing improvement in all key metrics.  
>
> **This improvement is due to correcting the input preprocessing:**  
> I removed the extra normalization layer and now use only `tf.keras.applications.resnet50.preprocess_input`, which ensures the data matches what ResNet-50 expects.

- Kaggle Notebook : https://www.kaggle.com/code/haro0n/encide-catvsdog

## 📊 Training Summary

- **Dataset:** ~10,000 images (train), 10,000 for validation, 5,000 test
- **Base model:** ResNet50 (ImageNet weights), frozen for 15 epochs, then top ~200 layers fine-tuned for 10 epochs  
- **Best validation accuracy:** **98.8%**  
- **Final validation metrics:**  
  - **Accuracy:** 98.8%  
  - **AUC:** 0.9993  
  - **Precision:** 0.9868  
  - **Recall:** 0.9890  
  - **Loss:** 0.0297  


![Training History](/logs/training_history.png)

> Full training log available in `logs/train.log


---
## Technologies & Dependencies

- Python >=3.12 (`pyproject.toml` specifies `requires-python = ">=3.12"`)
- TensorFlow >=2.19.0
- Matplotlib >=3.10.1
- [uv](https://github.com/astral-sh/uv) for dependency management and installation

---

## Workflow

1. **Data Loading & Preprocessing**
   - Images are loaded from directory structure using `tf.keras.utils.image_dataset_from_directory`.
   - Data augmentation includes random flips, rotations, zoom, contrast, and translation.
   - Normalization is applied to all images.

2. **Model Architectures**
   - Custom CNN: Multiple convolutional blocks with BatchNormalization and Dropout.
   - ResNet-50: Pretrained on ImageNet, with custom dense layers added. The base model is initially frozen, then partially unfrozen for fine-tuning.

3. **Training**
   - EarlyStopping and ModelCheckpoint are used to prevent overfitting and save the best model.
   - Learning rate scheduling is used for the custom CNN; a fixed low learning rate is used for fine-tuning ResNet-50.

4. **Evaluation & Visualization**
   - Training and validation metrics (accuracy, loss, AUC, precision, recall) are reported.
   - Training history is visualized and saved as `training_history.png`.

5. **Model Export**
   - Models are saved in Keras (`.keras`) and TensorFlow Lite (`.tflite`) formats for deployment or further use.

---

## Dataset Structure
  ```
   data/
   ├── train/
   │   ├── cats/
   │   └── dogs/
   ├── test/
   │   ├── cats/
   │   └── dogs/
   ```

---

## How to Run

1. **Clone the repository:**
    ```
    git clone https://github.com/haroon0x/encide-ml-haroon.git
    cd encide-ml-haroon
    ```

2. **Install dependencies using uv:**
    
    ```
    pip install uv
    ```

    ```
    uv pip install -r requirements.txt
    ```
    _or, if using uv’s project workflow:_
    ```
    uv venv
    uv pip install -r pyproject.toml
    ```

3. **Prepare the dataset** 
   ## 📥 Download Kaggle Data
   
   ```bash
   kaggle datasets download salader/dogs-vs-cats
   unzip dogs-vs-cats.zip -d data/

4.   ## ⚙️ Run Training

   ```bash
   python src/train.py \
   --train_dir data/train \
   --test_dir data/test \
   --batch_size 16 \
   --epochs_frozen 15 \
   --epochs_finetune 10 \
   --output_log logs/train.log

5. **Run the desired notebook or script** to train and evaluate a model.
   
## 📊 View Results

-  **Plot:** `logs/training_history.png`  
- **Console log:** `logs/train.log`

## Acknowledgements
- Dataset: [Salader’s Dogs vs. Cats](https://www.kaggle.com/salader/dogs-vs-cats)  
- Inspiration: [TensorFlow Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)  


---




