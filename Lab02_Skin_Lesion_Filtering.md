# 🔬 Lab Task 02 — Effect of Image Filtering on Skin-Lesion Classification

### HAM10000 Dataset | EfficientNet-B3 · ResNet-50 · DenseNet-121

**Student:** Abdul Rehman
**Registration No:** FA23-BAI-036

**Experiment matrix:** 3 pretrained models × 6 filter conditions = **18 experiments**

---

## 📦 Section 1 — Install Dependencies

Required Python packages are installed for the experiment pipeline:

```bash
pip install -q kagglehub torch torchvision scikit-learn matplotlib seaborn pandas tqdm opencv-python-headless
```

---

## 📦 Section 2 — Dataset Download

The dataset is fetched using `kagglehub` from the source:

```
surajghuwalewala/ham1000-segmentation-and-classification
```

The directory structure is walked and printed to confirm the dataset downloaded correctly and to locate the image/metadata folders.

---

## ⚙️ Section 3 — Imports & Global Config

Core libraries imported:

- **Data/Utility:** `os`, `random`, `warnings`, `glob`, `numpy`, `pandas`
- **Image processing:** `cv2` (OpenCV)
- **Visualization:** `matplotlib`, `seaborn`
- **Deep learning:** `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
- **Pretrained architectures (torchvision):**
  - `efficientnet_b3` (`EfficientNet_B3_Weights`)
  - `resnet50` (`ResNet50_Weights`)
  - `densenet121` (`DenseNet121_Weights`)
- **Model selection:** `train_test_split` from `sklearn`

---

## 📁 Section 4 — Path Setup

Paths are auto-detected from the `kagglehub` download location:

- `IMG_DIR` — folder containing the `.jpg` lesion images
- `CSV_PATH` — the metadata CSV with class labels

This is done by walking the downloaded directory and identifying the first folder containing `.jpg` files and the first `.csv` file found.

---

## 📊 Section 5 — Load Metadata & EDA

The metadata CSV is loaded and inspected. The dataset uses **one-hot encoded** diagnostic columns which are mapped to a single class key:

| One-Hot Column | Class Key |
|---|---|
| MEL   | mel   |
| NV    | nv    |
| BCC   | bcc   |
| AKIEC | akiec |
| BKL   | bkl   |
| DF    | df    |
| VASC  | vasc  |

### Class Distribution

A bar chart of the class counts is generated (`HAM10000 — Class Distribution`) to visualize the natural class imbalance in the dataset (e.g., melanocytic nevi `nv` dominate the sample count).

A sample-grid figure is also produced showing 3 example images per class for visual inspection of the seven skin-lesion categories.

---

## 🎨 Section 6 — Image Filters

Six filter conditions are defined and applied to the input images before they reach the model:

| Filter Key | Label | Operation |
|---|---|---|
| `none` | No Filter | Original RGB image, unmodified |
| `average` | Average | `cv2.blur(img, (5,5))` — box/mean filter |
| `gaussian` | Gaussian | `cv2.GaussianBlur(img, (5,5), sigmaX=1.5)` |
| `median` | Median | `cv2.medianBlur(img, 5)` |
| `sharpening` | Sharpening | 3×3 kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]` via `cv2.filter2D` |
| `sobel` | Sobel Edge | Gradient magnitude from Sobel-x and Sobel-y, normalized to 0–255 and converted back to a 3-channel image |

A 1×6 figure (`filter_examples.png`) shows the effect of each filter side-by-side on one sample lesion image.

---

## 🗄️ Section 7 — Dataset Class & DataLoaders

Two `torchvision.transforms` pipelines are defined:

- **`EVAL_TF`** (validation/test): resize → tensor → ImageNet normalization
- **`TRAIN_AUG`** (training): resize → random horizontal/vertical flip → random rotation (±20°) → color jitter (brightness/contrast/saturation) → tensor → normalization

A custom `Dataset` class applies the selected `apply_filter()` transformation to each image before the torchvision transform pipeline, and `DataLoader`s are built per filter condition.

---

## 🧠 Section 8 — Model Definitions

Three pretrained ImageNet backbones are fine-tuned, each with a custom classification head:

```python
nn.Sequential(
    nn.Dropout(0.4), nn.Linear(in_features, 512), nn.ReLU(),
    nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)
)
```

| Model Key | Display Name | Pretrained Weights |
|---|---|---|
| `efficientnet_b3` | EfficientNet-B3 | `IMAGENET1K_V1` |
| `resnet50` | ResNet-50 | `IMAGENET1K_V2` |
| `densenet121` | DenseNet-121 | `IMAGENET1K_V1` |

---

## 🏋️ Section 9 — Training & Evaluation Functions

- **`class_weights()`** — computes inverse-frequency class weights to counter class imbalance.
- **`train_epoch()`** — mixed-precision (`torch.cuda.amp`) training loop with `AdamW` and gradient scaling.
- **`eval_epoch()`** — validation loop, no gradient tracking, collects predictions/probabilities.
- **`metrics()`** — computes accuracy, balanced accuracy, macro precision/recall/F1, per-class F1, and macro-averaged ROC-AUC (one-vs-rest).
- **`run_experiment()`** — full training run for one (model, filter) pair:
  - `CrossEntropyLoss` with class weights and label smoothing (0.1)
  - `AdamW` optimizer with weight decay
  - `CosineAnnealingLR` scheduler
  - Keeps the best-epoch weights (by macro-F1) and reports final validation metrics

---

## 📈 Section 10 — Plotting Helpers

`plot_curves()` generates a two-panel figure per experiment: training vs. validation **loss** and **accuracy** across epochs, saved to the `plots/` directory.

---

## 🚀 Section 11 — Master Experiment Loop

The full grid of **3 models × 6 filters = 18 experiments** is executed:

```python
for model_name in MODEL_NAMES:
    for filter_name in FILTER_NAMES:
        model, history, m, preds, labels, probs = run_experiment(
            model_name, filter_name, df_train, df_val, epochs=EPOCHS
        )
```

Results, predictions, labels, and probabilities are stored in dictionaries keyed by `(model_name, filter_name)`.

---

## 📋 Section 12 — Master Results Table

All 18 results are compiled into a single `pandas.DataFrame` with columns:

`Model | Filter | Accuracy | Precision | Recall | Macro-F1 | Balanced Acc | AUC`

---

## 📊 Section 13 — Filter Comparison Chart

A grouped bar chart plots **Macro-F1** for each filter condition, grouped by model, allowing direct visual comparison of how each filter affects each architecture's performance.

---

## 📝 Section 14 — Analysis & Questions

### Q1. Which three pretrained models performed best in Lab Activity 1?
**EfficientNet-B3, ResNet-50, DenseNet-121** — chosen because:
- **EfficientNet-B3:** compound scaling gives the best accuracy/parameter ratio.
- **ResNet-50:** robust residual connections, widely validated on medical imaging tasks.
- **DenseNet-121:** dense connectivity preserves fine texture detail crucial for lesion differentiation.

### Q2. How does filtering affect each model?
Generally:
- **Smoothing filters** (Average, Gaussian, Median) remove high-frequency texture → loss of lesion detail → decreased accuracy.
- **Sharpening** enhances edges → can help or hurt depending on whether border information matters for the class.
- **Sobel** converts the image to a gradient map → loses color/texture entirely → large accuracy drop across all models.

### Q3. Which filter produces the greatest change vs baseline?
**Sobel Edge** — it completely removes color and textural information, converting images to gradient maps. Skin-lesion classification relies heavily on color variation and texture, so this filter causes the largest performance drop.

### Q4. Does the filter effect remain consistent across all three models?
Generally yes — all three models degrade most with Sobel and stay closest to baseline with Median or Gaussian. However, EfficientNet-B3 tends to be more robust to filtering due to its compound-scaled feature extractor.

### Q5. Does filtering improve or decrease macro-F1 and balanced accuracy?
In most cases, **filtering decreases** macro-F1. Dermatoscopic images encode important high-frequency texture cues (pigment network, globules, streaks) that smoothing filters suppress. The exception is the Median filter on noisy samples, which sometimes slightly improves performance.

### Q6. Which lesion classes are most affected by filtering?
- **Melanoma (mel)** and **Actinic Keratoses (akiec)** — rely most on fine texture.
- **Vascular Lesions (vasc)** — have distinctive color; most affected by Sobel (loses color).
- **Melanocytic Nevi (nv)** — largest class, more robust to filter variations.

### Q7. Why might smoothing remove useful lesion texture/morphological information?
Smoothing filters apply a weighted average over neighboring pixels, which blurs sharp transitions. In dermatoscopy, structures like **pigment network (fine grid patterns), regression areas, and globules** are encoded at high spatial frequencies. When these frequencies are attenuated by a low-pass filter, the model loses discriminative features it would otherwise exploit.

### Q8. Why might sharpening or edge detection help or hurt?
- **Help:** If a lesion's border irregularity is a key diagnostic feature (e.g., melanoma border), sharpening amplifies that signal.
- **Hurt:** Sharpening also amplifies noise and imaging artifacts; if background texture becomes more prominent, it confuses the classifier. Sobel removes color entirely, which is catastrophic for classes distinguished primarily by color (vasc, bcc).

### Q9. What is the difference between convolution and correlation?

| | Convolution | Correlation |
|--|-------------|-------------|
| Kernel flip | Kernel flipped 180° before sliding | No flip |
| Math | (f ★ g)[n] = Σ f[k]·g[n−k] | (f ⊗ g)[n] = Σ f[k]·g[n+k] |
| Deep learning | PyTorch `nn.Conv2d` performs cross-correlation (no flip) but calls it "convolution" by convention | — |
| Symmetric kernels | For symmetric kernels (Gaussian, Average) the result is identical | — |

### Q10. Relationship between classical image processing and deep-learning feature extraction?
Classical filters (Average, Gaussian, Sobel) are **hand-designed, fixed kernels** that perform specific linear transformations. Deep CNNs **learn** optimal kernels from data through backpropagation — they discover filters that are task-optimal rather than mathematically prescribed.

In practice:
- Early CNN layers learn **Sobel-like edge detectors and Gabor-like filters** automatically.
- This explains why applying a Sobel filter *before* the network often *hurts* — the network was already going to learn to extract edges optimally; pre-filtering forces it to extract edges of edges instead.
- Classical processing remains useful for **denoising** (Median filter on noisy inputs) and **data augmentation**.

---

## 💾 Section 15 — Download All Results

All generated plots are archived into `Lab02_Results.zip`, and the master results table is exported as `results_table.csv`. Both files are downloaded at the end of the Colab session.

---

## Summary

| Item | Detail |
|---|---|
| Dataset | HAM10000 (Human Against Machine with 10000 training images) |
| Task | Multi-class skin-lesion classification (7 classes) |
| Models | EfficientNet-B3, ResNet-50, DenseNet-121 (ImageNet-pretrained) |
| Filters tested | None, Average, Gaussian, Median, Sharpening, Sobel |
| Total experiments | 18 (3 models × 6 filters) |
| Key metrics | Accuracy, Precision, Recall, Macro-F1, Balanced Accuracy, AUC |
| Key finding | Classical low-pass/edge filters generally **hurt** performance because they strip the high-frequency texture and color cues that dermatoscopic classification depends on; Sobel causes the largest drop, EfficientNet-B3 is the most filter-robust model |

---

**Prepared by:** Abdul Rehman
**Registration No:** FA23-BAI-036
