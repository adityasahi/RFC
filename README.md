# # Cancer Classification and Clustering Pipeline

This project provides a comparative analysis of machine learning models for binary classification of cancer types, alongside clustering using KMeans. It includes preprocessing, feature selection, dimensionality reduction, and model evaluation with both KNN and Random Forest classifiers.

---

## 📂 Dataset

The dataset used is assumed to be in `dataset.csv`, where:
- The first column (`Unnamed: 0`) contains sample IDs, with cancer type embedded in their names.
- Remaining columns are features (gene expressions or similar metrics).

---

## 🧪 Methods

### 1. Preprocessing
- Label extraction from sample names (e.g., "brca", "luad", "prad").
- Scaling using `StandardScaler`.
- Feature selection using `SelectKBest` (top 100 features).

### 2. Classification Models
| Model             | Notes |
|------------------|-------|
| K-Nearest Neighbors (KNN) | Tuned using GridSearchCV |
| **Random Forest**         | Newly added; grid-searched for optimal hyperparameters |

### 3. Clustering
- Unsupervised classification using KMeans (k=2).
- Dimensionality reduction via PCA (for visualization).
- Clustering accuracy calculated by label inversion matching.

---

## 📊 Cancer Type Comparisons

This pipeline performs pairwise comparisons:
- `brca` vs `luad`
- `brca` vs `prad`
- `luad` vs `prad`

Each pair undergoes:
- Clustering visualization
- Training and evaluation of both KNN and Random Forest
- Per-class accuracy analysis

---

## 📈 Output

- Confusion Matrices (Random Forest) saved as PNGs.
- Classification reports printed to console.
- Accuracy scores for both models and all comparisons.

---

