# HAM10000 Skin Lesion Classification

A deep learning project for classifying skin lesions using the **HAM10000 dataset**. The project compares multiple pretrained CNN architectures and evaluates their performance using several classification metrics.

## Project Overview

This project uses transfer learning with different deep learning models to classify skin lesion images into **7 diagnostic classes**.

The HAM10000 dataset contains **10,015 skin lesion images** belonging to the following classes:

* `akiec`
* `bcc`
* `bkl`
* `df`
* `mel`
* `nv`
* `vasc`

The notebook performs dataset preparation, image preprocessing, model training, evaluation, and performance comparison.

## Models Used

The following 8 deep learning models were trained and evaluated:

1. AlexNet
2. VGG16
3. VGG19
4. ResNet18
5. ResNet50
6. ResNet101
7. DenseNet121
8. EfficientNet-B0

## Dataset Split

The dataset was divided into:

* **Training:** 5,229 images
* **Validation:** 747 images
* **Testing:** 1,494 images

Stratified splitting was used to maintain class distribution.

## Image Preprocessing

Images are resized to **224 × 224** pixels.

Training images use augmentation techniques including:

* Random cropping
* Horizontal flipping
* Vertical flipping
* Random rotation
* Color jitter
* Normalization

A weighted random sampler is also used to help handle class imbalance.

## Training Method

Each model uses a two-phase training strategy:

### Phase 1

* 4 epochs
* Learning rate: `1e-3`
* Warm-up of the classifier and selected final layers

### Phase 2

* 6 epochs
* Learning rate: `1e-5`
* Fine-tuning of the model

Mixed-precision training is used to reduce GPU memory usage.

The notebook also saves trained model checkpoints and results to Google Drive and can resume training after a Colab session disconnects.

## Results

The best-performing model in the reported experiment was **ResNet101**.

| Model           | Test Accuracy | Precision |     Recall | F1 Score |    AUC |
| --------------- | ------------: | --------: | ---------: | -------: | -----: |
| ResNet18        |        75.37% |    83.64% |     75.37% |   78.14% | 94.84% |
| ResNet50        |        79.12% |    84.25% |     79.12% |   80.93% | 95.34% |
| ResNet101       |    **79.18%** |    83.36% | **79.18%** |   80.81% | 94.61% |
| DenseNet121     |        77.24% |    83.05% |     77.24% |   79.28% | 94.53% |
| EfficientNet-B0 |        75.50% |    82.43% |     75.50% |   77.98% | 93.71% |

Additional model results are available in the notebook and generated result tables.

## Technologies Used

* Python
* PyTorch
* Torchvision
* TIMM
* Scikit-learn
* XGBoost
* NumPy
* Pandas
* PIL
* Matplotlib
* Google Colab
* Google Drive
* Kaggle HAM10000 Dataset

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Model efficiency is also measured using parameters, model size, FLOPs, and inference time.

## Files

```text
HAM10000-Skin-Lesion-Classification/
│
├── lab_task_(1).ipynb
├── README.md
└── table.csv
```

## How to Run

### 1. Open the notebook

Open `lab_task_(1).ipynb` in **Google Colab**.

### 2. Install dependencies

The notebook installs the required packages automatically.

### 3. Mount Google Drive

The notebook uses Google Drive to save model checkpoints and results.

### 4. Download the dataset

The HAM10000 dataset is downloaded using `kagglehub`.

### 5. Run the notebook

Run the cells in order to:

1. Install dependencies
2. Mount Google Drive
3. Load libraries and configuration
4. Download the HAM10000 dataset
5. Prepare and split the data
6. Apply image transformations
7. Create the models
8. Train and evaluate all models
9. Generate comparison tables
10. Save model checkpoints and results

## Important Note

This project is intended for **research and educational purposes**. It should not be used as a substitute for professional medical diagnosis or clinical decision-making.

## Author

**Abdul Rehman**

Artificial Intelligence Student
COMSATS University Islamabad, Wah Campus
