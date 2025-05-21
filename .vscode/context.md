# Apple Scab Detection System: Pipeline Documentation

## 📂 Dataset Structure (Only Two Classes)

```
dataset/apple/
├── apple scab/
├── healthy/
```

Each subfolder contains RGB `.jpg` or `.png` images.

---

## 🧱️ Pipeline Summary

### 0. Environment

- **OS**: Linux
- **Language**: Python 3.x
- **Virtual Environment**: `venv`
- **Allowed Libraries**: `numpy`, `pandas`, `matplotlib`
- **Forbidden**: Any image processing or ML libraries (e.g., OpenCV, PIL, scikit-learn)

---

### 1. Preprocessing

#### 🔄 Steps:

- **Image Loading**

  - Use `matplotlib.image.imread()` to read RGB image into NumPy array.
  - Ensure shape = `(H, W, 3)` with values in `[0, 255]`.

- **Cropping via Background Removal**

  - Background = R > 220 and G > 220 and B > 220
  - Create a binary mask of non-background pixels
  - Compute bounding box (min and max row/col) of mask
  - Crop original RGB image to bounding box

- **RGB to HSV Conversion (Manual)**

  - Implement RGB→HSV using standard formulas (vectorized in NumPy)
  - Output:

    - H: \[0, 360] (or normalized to \[0.0–1.0])
    - S: \[0.0–1.0]
    - V: \[0.0–1.0]

#### 🔍 Preprocessing Preview (Visualization)

For sample image, save a **side-by-side preview** including:

- Original image
- Cropped image
- HSV Saturation channel (grayscale)
- Mask showing V < 0.2 areas (black spots)

**Save to**:

```
visuals/previews/img_<filename>_preview.png
```

---

### 2. Feature Extraction

From each processed image, extract:

- **mean_saturation**: Mean of Saturation (S) channel
- **black_spot_ratio**: Fraction of pixels where V < 0.2

**Result Format (CSV)**:

```csv
filename,label,mean_saturation,black_spot_ratio
leaf_001.jpg,apple scab,0.43,0.18
leaf_002.jpg,healthy,0.52,0.03
...
```

**Save all extracted features to**:

```
output/features_YYYY-MM-DD_HH-MM.csv
```

---

### 📊 Feature Preview (Before Classification)

Create a **2D scatter plot** of extracted features:

- X-axis: mean_saturation
- Y-axis: black_spot_ratio
- Color-code by class (apple scab vs healthy)
- Label with legends and axis titles

**Save to**:

```
visuals/features/feature_space_<timestamp>.png
```

Purpose: to inspect feature separability before training.

---

### 3. SVM Classifier (From Scratch)

Implement binary linear SVM from scratch using NumPy only.

#### ⚙️ Requirements:

- Use primal form (hard or soft margin)
- Optimization: gradient or subgradient descent
- Support hyperparameters:

  - learning rate
  - regularization (C)
  - number of epochs

- Normalize input features before training
- Implement `train(X, y)` and `predict(X)` methods

#### 📈 SVM Visualization

- Plot **training loss vs epoch**
- Plot **decision boundary** on 2D feature space (with points and margins)
- Highlight support vectors if applicable

**Save to**:

```
visuals/svm/decision_boundary_<timestamp>.png
visuals/svm/loss_curve_<timestamp>.png
```

---

### 4. Evaluation (Optional)

After training:

- Evaluate: accuracy, precision, recall, F1-score
- Visualize: 2D decision boundary on test data, confusion matrix

**Save evaluation outputs to**:

```
output/evaluation_<timestamp>.csv
visuals/eval/confusion_matrix_<timestamp>.png
```

---

## 🔄 Post-Training Comparison Preview

Create a **side-by-side comparison plot**:

- Left: feature scatter plot **before training**
- Right: feature scatter plot with **SVM decision boundary**

**Save to**:

```
visuals/comparisons/pre_vs_post_svm_<timestamp>.png
```

Purpose: visualize classifier effect on the raw feature space.

---

## 📌 System Constraints

- Only process `"apple scab"` and `"healthy"` classes
- No pretrained models or external ML/CV libraries
- Image processing and feature engineering must be **deterministic and reproducible**
- Timestamp all saved outputs for version tracking
- Modular structure in `src/` for each stage

---

## 🎯 Main Objective

Build a lightweight, explainable, and accurate apple disease detection pipeline using:

- Manual preprocessing
- Simple yet discriminative features
- Custom SVM classifier
- Clear visual outputs at every stage

Maintain reproducibility, modularity, and clarity throughout the pipeline.
