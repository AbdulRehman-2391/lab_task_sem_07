# Blood Cell Classification Using HOG and SIFT

### Computer Vision Lab Mid Exam

This project compares two classical computer vision techniques, **Histogram of Oriented Gradients (HOG)** and **Scale-Invariant Feature Transform (SIFT)**, for blood cell classification using Support Vector Machines (SVM).

The main objective is to evaluate the classification performance and robustness of both feature extraction techniques under image rotation and Gaussian noise.

## Project Information

- **Student:** Abdul Rehman
- **Registration Number:** FA23-BAI-036
- **University:** COMSATS University Islamabad, Wah Campus
- **Course:** Computer Vision Lab
- **Project:** Lab Mid Exam — HOG vs SIFT

## Dataset

The project uses the **BloodMNIST** dataset from the MedMNIST collection.

Four blood cell classes are selected:

1. Eosinophil (Class 1)
2. Monocyte (Class 5)
3. Neutrophil (Class 6)
4. Platelet (Class 7)

### Dataset Distribution

| Dataset Split | Images per Class | Total Images |
|---|---:|---:|
| Training | 100 | 400 |
| Validation | 30 | 120 |
| Testing | 30 | 120 |
| **Total** | **160** | **640** |

All images are converted to grayscale and resized to **128 × 128 pixels**.

## Technologies Used

- Python
- Google Colab
- OpenCV
- Scikit-learn
- Scikit-image
- NumPy
- Matplotlib
- MedMNIST

## Methodology

### 1. HOG + SVM

Histogram of Oriented Gradients extracts image features based on gradient directions and edge structures.

**Configuration:**
- Cell size: 8 × 8
- Orientations: 9
- Cells per block: 2 × 2
- Feature vector length: 8,100
- Classifier: LinearSVC

### 2. SIFT + SVM

Scale-Invariant Feature Transform extracts local image features using keypoints and descriptors.

Normal SIFT detection is first performed to analyze the number of detected keypoints.

A fixed-grid SIFT approach is then implemented to produce consistent feature vector lengths for SVM classification.

**Configuration:**
- Grid size: 6 × 6
- Total keypoints: 36
- Keypoint size: 32
- Descriptor length: 128
- Feature vector length: 4,608
- Classifier: LinearSVC

## Model Evaluation

Both models are evaluated using:

- Validation accuracy
- Macro-F1 score
- Confusion matrix
- Classification accuracy under rotation and noise

### Robustness Testing

Three testing conditions are considered:

1. **Clean images:** Original test images.
2. **Rotated images:** Images rotated by 20 degrees.
3. **Noisy images:** Gaussian noise with sigma = 0.05.

### Experimental Results

The following results were obtained from the initial notebook execution:

| Model | Clean Accuracy | Rotated Accuracy | Noisy Accuracy |
|---|---:|---:|---:|
| HOG + SVM | 71.67% | 70.00% | 25.00% |
| SIFT + SVM | 70.83% | 53.33% | 71.67% |

**Note:** These results are from the initial experiment. The SVM hyperparameter selection must be corrected to use only C = {0.1, 1, 10}, as specified in the lab instructions. The final results should be updated after rerunning the notebook.

## Key Findings

- HOG achieved slightly higher classification accuracy on clean test images.
- HOG performed better than SIFT under 20-degree rotation.
- SIFT demonstrated significantly better robustness against Gaussian noise.
- Both approaches successfully classified blood cell images using classical machine learning techniques.

## How to Run

1. Open Google Colab.
2. Upload the Jupyter Notebook (`.ipynb`).
3. Install the required libraries:

```python
!pip install medmnist opencv-python scikit-image scikit-learn matplotlib
```

4. Run all notebook cells sequentially.
5. Review the generated visualizations, confusion matrices, and classification results.

## Project Structure

```text
Blood-Cell-Classification/
│
├── FA23_BAI_036_AbdulRehman_LabMid.ipynb
├── README.md
└── LabMid_Report.pdf
```

## Conclusion

This project demonstrates how classical feature extraction techniques can be combined with Support Vector Machines for blood cell classification.

The initial experimental results indicate that HOG performs better under rotation, while fixed-grid SIFT provides greater resistance to Gaussian noise.

The comparison highlights the importance of selecting appropriate feature extraction techniques based on the expected image conditions.

## Author

**Abdul Rehman**  
BS Artificial Intelligence  
COMSATS University Islamabad, Wah Campus

**Registration Number:** FA23-BAI-036
