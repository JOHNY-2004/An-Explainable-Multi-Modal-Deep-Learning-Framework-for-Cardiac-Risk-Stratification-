# 🫀 An Explainable Multi-Modal Deep Learning Framework for Cardiac Risk Stratification

### 🧩 Overview

This project presents an **explainable AI pipeline** for predicting patient-specific cardiac risk levels by integrating **cardiac MRI images** and **structured clinical data**.
It combines deep learning–based **myocardial segmentation**, **radiomics feature extraction**, and **multi-modal risk prediction** using **XGBoost**, **Attention-based MLP**, and **SHAP + Grad-CAM explainability**.

---

## 🚀 Key Features

* **Multi-Modal Fusion:** Combines radiomics features from cardiac MRI with clinical biomarkers (LVEF, NT-proBNP, Troponin, Age).
* **Automated Myocardium Segmentation:** Trained U-Net model identifies myocardium from short-axis MRI slices.
* **Radiomics Extraction:** Quantitative shape, texture, and intensity features computed using *PyRadiomics*.
* **Risk Prediction Models:**

  * XGBoost Classifier for tabular data.
  * Attention-based Dual-Branch MLP for deep feature learning.
  * Stacked Ensemble (Logistic Regression meta-learner) for final calibrated output.
* **Explainability:**

  * **Grad-CAM** – visual heatmaps highlighting myocardial regions driving predictions.
  * **SHAP** – feature-level explanations showing how clinical and radiomic features influence each prediction.
* **Result Output:** Predicts risk class → *Low, Moderate, High, Very High* with visual & quantitative explanations.

---

## ⚙️ Implementation Workflow

1. **Data Preparation**

   * Load cardiac MRI & clinical CSV data.
   * Preprocess images and normalize clinical variables.

2. **Segmentation (U-Net)**

   * Train on EMIDEC dataset; output predicted myocardium masks.

3. **Feature Extraction**

   * Use *PyRadiomics* to extract texture, shape, and first-order features.
   * Merge with clinical parameters into `combined_radiomics_features.csv`.

4. **Model Training**

   * Balance data using SMOTE + RandomUnderSampler.
   * Train XGBoost & Attention-MLP; fuse via Stacked Ensemble.

5. **Explainability Analysis**

   * **SHAP:** global feature importance & per-patient waterfall plots.
   * **Grad-CAM:** visualize MRI regions influencing predictions.

6. **Output**

   * Predicted risk category (Low / Moderate / High / Very High).
   * Visual explanations (Grad-CAM + SHAP).

---

## 🧩 Explainability Modules

### 🧠 SHAP (SHapley Additive exPlanations)

* Quantifies the impact of each input feature on model output.
* Provides both **global** (overall feature influence) and **local** (individual patient reasoning) interpretability.
* Example:

  * **High NT-ProBNP** and **Low LVEF** push the prediction toward **High/Very High Risk**.
  * **Normal Troponin and LVEF** lower the predicted risk level.

### 🫀 Grad-CAM (Gradient-weighted Class Activation Mapping)

* Highlights the **specific myocardial regions** that most influenced the model’s decision.
* Produces a color heatmap overlay (red = high influence, blue = low influence).
* Enables clinicians to **verify the anatomical areas** responsible for the prediction, improving trust.

Together, **SHAP + Grad-CAM** make the system *transparent, interpretable, and clinically reliable*.

---

## 🧭 Future Improvements

* Expand to **larger, multi-center MRI datasets** for stronger generalization.
* Incorporate **transformer-based fusion models** for cross-modal learning.
* Develop a **web-based clinical dashboard** for real-time risk evaluation.
* Integrate **federated learning** for privacy-preserving multi-hospital collaboration.
* Introduce **auto-ML pipelines** for automated hyperparameter optimization and retraining.

---

## 👥 Authors

**22BCE2358 – Peethala Hamal Johny**
**22BCE3815 – Vaka Abhilesh**


---
