# Blood Cell Image Classification

Classifying eight types of peripheral blood cells from microscope images, comparing 15 models: classic machine learning on hand-crafted features, a CNN trained from scratch, and transfer learning with MobileNetV2 (frozen and fine-tuned).

**Best model: fine-tuned MobileNetV2 — 95.9% accuracy, 0.957 macro F1, 0.998 macro AUC on 3,415 held-out test images.**

![Eight blood-cell categories](images/00_cell_types_samples.png)

> Educational project. Not a diagnostic tool.

---

## Results

| Metric (test set, 3,415 images) | Value |
|---|---|
| Accuracy | 95.90% (3,275 correct) |
| Macro precision | 0.9579 |
| Macro recall | 0.9565 |
| Macro F1 | 0.9572 |
| Macro AUC | 0.9982 |

### Model comparison (validation set, ranked by macro F1)

| Model | Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| **MobileNetV2 — fine-tuned** | **0.9682** | **0.9667** | **0.9986** |
| XGBoost (300 features) | 0.9480 | 0.9445 | 0.9969 |
| Small CNN (from scratch) | 0.9458 | 0.9432 | 0.9970 |
| MobileNetV2 — frozen base | 0.9396 | 0.9351 | 0.9967 |
| SVM — polynomial | 0.9250 | 0.9181 | 0.9857 |
| … 10 more classic models | | | |

![Model comparison table](images/01_model_comparison_table.png)

Only the selected model (fine-tuned MobileNetV2) was evaluated on the test set.

![Combined ROC curves](images/02_combined_roc_all_models.png)

### Where the model makes mistakes

The errors cluster among **immature granulocytes (ig)**, **monocytes** and **neutrophils**:

| Class | Test recall |
|---|---|
| eosinophil | 99.68% |
| platelet | 99.36% |
| lymphocyte | 97.53% |
| basophil | 96.28% |
| neutrophil | 95.95% |
| erythroblast | 93.55% |
| ig | 91.70% |
| monocyte | 91.20% |

Largest confusions: monocyte → ig (21), neutrophil → ig (18), ig → monocyte (16), ig → neutrophil (13).

![Test confusion matrices](images/03_test_confusion_matrices.png)

### What the model looks at (Grad-CAM)

Heat-maps show that, for correct predictions, the network focuses on the cell itself, not the background. Grad-CAM shows where the model looked, not why it chose a class.

![Grad-CAM](images/04_gradcam_correct_vs_mistakes.png)

![Final test metrics](images/05_final_test_metrics_report.png)

---

## Workflow

1. **Data checks** — 17,093 images; removed 1 unreadable file, 16 exact duplicate copies and 2 conflicting-label images → 17,074 clean images.
2. **Stratified split (seed 42)** — 64% train (10,927), 16% validation (2,732), 20% test (3,415). Test set held out until final evaluation.
3. **Hand-crafted features** — 1,828 per image: RGB statistics, HSV colour histograms, LBP texture and HOG edges. Scaling and feature selection (top 300 by ANOVA F-score) fitted on training data only.
4. **12 classic models** — Logistic Regression, SVM (RBF / polynomial / linear), Random Forest, XGBoost, KNN, Decision Tree, Bagging, Extra Trees, AdaBoost, MLP.
5. **Small CNN** — three convolutional blocks trained from scratch with random flips and dropout.
6. **Transfer learning** — MobileNetV2 (ImageNet weights): first a frozen base with a new classification head, then fine-tuning from `block_13_expand` onward (learning rate 1e-5, batch-norm layers kept frozen).
7. **Cross-validation** — training-only 5-fold CV comparing 100 / 300 / 600 / all features for XGBoost and polynomial SVM (best: XGBoost at 600 features, F1 0.9450 ± 0.0067).
8. **Final evaluation** — per-class metrics, ROC/AUC, confusion matrices and Grad-CAM on the held-out test set.

## Limitations

- One dataset from a single source; performance on other labs, stains or microscopes is unknown.
- No large hyperparameter search (skipped for runtime); models use fixed settings.
- The model comparison uses a single validation split, so small gaps (e.g. XGBoost vs the small CNN) are not firm differences. Only XGBoost and the polynomial SVM were cross-validated.
- Neural-network results vary slightly between reruns (GPU non-determinism). The pipeline was rerun once with no setting changes: test accuracy moved from 95.70% to 95.90%. The numbers above are from the final run.
- `ig` is a catch-all class of immature cells that look like several other types.
- No medical validation.

## Repository contents

```
├── blood-cell-classification.ipynb   # full notebook with outputs
├── README.md
├── requirements.txt
└── images/                           # figures used in this README
```

## How to run

The notebook was developed on **Kaggle** (GPU) and reads the dataset from `/kaggle/input/...` and writes to `/kaggle/working/...`.

1. Create a Kaggle notebook and attach the dataset below.
2. Attach the MobileNetV2 ImageNet weights file `mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5` as an input dataset (the notebook loads it locally, without an internet download).
3. Upload `blood-cell-classification.ipynb` and run all cells.

To run elsewhere, change the dataset and output paths in the first cells. Dependencies are listed in `requirements.txt` (TensorFlow 2.20 was used).

## Dataset

Peripheral blood cell microscopy images (8 classes: basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil, platelet), accessed through Kaggle (`orvile/microscopic-peripheral-blood-cell-images`).
Please cite the original authors as listed on the dataset page — Acevedo et al., *Data in Brief*, 2020.

## Tech stack

Python · TensorFlow / Keras · scikit-learn · XGBoost · scikit-image · pandas · matplotlib
