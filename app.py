import os
import io
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

import shap

# MRI/Segmentation utilities
import nibabel as nib
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras import models, layers
import warnings as _w

# Optional torch (for stacked DL branch)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# -----------------------------
# Constants and file paths
# -----------------------------
CLINICAL_COLS = ["Age", "LVEF", "Troponin", "NTProBNP"]
CSV_PATH = "combined_radiomics_features.csv"
CLINICAL_MODEL_PATH = "clinical_model.pkl"
UNET_MODEL_PATH = "unet_multiclass.h5"
STACKED_PATH = "stacked_fusion_smote.pkl"
BEST_XGB_PATH = "best_prognostic_model.pkl"


# -----------------------------
# Utility: Rule-based reasoning
# -----------------------------
def classify_patient_risk_rule_based(age: float, lvef: float, troponin: float, ntprobnp: float):
    troponin_ng_L = troponin * 1000 if troponin < 10 else troponin
    if troponin_ng_L > 50:
        return 'Very High Risk (Acute Cardiac Event)', f"Acutely elevated Troponin ({troponin}) indicates myocardial injury."
    if lvef is not None and lvef <= 40:
        return 'High Risk (Chronic Heart Failure)', f"LVEF is {lvef} (<= 40%), indicating severely reduced function."
    is_high_ntprobnp = False
    if age > 75 and ntprobnp > 1800: is_high_ntprobnp = True
    elif age >= 50 and ntprobnp > 900: is_high_ntprobnp = True
    elif age < 50 and ntprobnp > 450: is_high_ntprobnp = True
    if is_high_ntprobnp:
        return 'High Risk (Chronic Heart Failure)', f"NT-proBNP of {ntprobnp} is very high for age {age}, indicating severe heart stress."
    if lvef is not None and 41 <= lvef <= 54:
        return 'Moderate Risk', f"LVEF is {lvef} (41-54%), indicating mildly reduced function."
    is_moderate_ntprobnp = False
    if age > 75 and 125 < ntprobnp <= 1800: is_moderate_ntprobnp = True
    elif age >= 50 and 125 < ntprobnp <= 900: is_moderate_ntprobnp = True
    elif age < 50 and 125 < ntprobnp <= 450: is_moderate_ntprobnp = True
    if is_moderate_ntprobnp:
        return 'Moderate Risk', f"NT-proBNP of {ntprobnp} is moderately elevated for age {age}."
    return 'Low Risk', f"LVEF ({lvef}), Troponin ({troponin}), and NT-proBNP ({ntprobnp}) are within normal limits for age {age}."


# -----------------------------
# Clinical-only model (train/load)
# -----------------------------
def load_or_train_clinical_model(csv_path: str = CSV_PATH, model_path: str = CLINICAL_MODEL_PATH):
    if os.path.exists(model_path):
        return joblib.load(model_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training data CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # Prefer Risk_Score if numeric; else use Risk_Category
    y_series = None
    if 'Risk_Score' in df.columns:
        y_series = df['Risk_Score']
    elif 'Risk_Category' in df.columns:
        y_series = df['Risk_Category']
    else:
        raise ValueError("CSV must contain 'Risk_Score' or 'Risk_Category'")

    X = df[CLINICAL_COLS].copy().astype(float)
    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, multi_class='auto')
    model.fit(X_scaled, y)

    assets = {
        "model": model,
        "scaler": scaler,
        "classes": list(le.classes_),
        "label_encoder": le
    }
    joblib.dump(assets, model_path)
    return assets


def predict_clinical_model(assets, age: float, lvef: float, troponin: float, ntprobnp: float):
    X = np.array([[age, lvef, troponin, ntprobnp]], dtype=float)
    Xs = assets["scaler"].transform(X)
    probs = assets["model"].predict_proba(Xs)[0]
    idx = int(np.argmax(probs))
    return assets["classes"][idx], probs, idx


def shap_explain_instance(assets, age: float, lvef: float, troponin: float, ntprobnp: float):
    X = np.array([[age, lvef, troponin, ntprobnp]], dtype=float)
    Xs = assets["scaler"].transform(X)
    model = assets["model"]

    # Build a small, valid background in the scaled space
    bg_orig = np.array([
        [60, 55, 10, 300],
        [70, 45, 20, 900],
        [50, 60, 5, 150],
        [65, 50, 15, 600]
    ], dtype=float)
    bg_scaled = assets["scaler"].transform(bg_orig)

    # Use the unified API to avoid shape issues
    explainer = shap.Explainer(model.predict_proba, bg_scaled)
    exp = explainer(Xs)

    # Pick predicted class from multi-output tensor
    probs = model.predict_proba(Xs)[0]
    pred_idx = int(np.argmax(probs))
    # exp.values shape: (n_samples, n_features, n_outputs) for multiclass
    vals = np.array(exp.values)
    if vals.ndim == 3 and vals.shape[0] >= 1:
        sv = vals[0, :, pred_idx]
    elif vals.ndim == 2:
        sv = vals[0]
    else:
        sv = vals.squeeze()

    feature_names = CLINICAL_COLS
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "value": [age, lvef, troponin, ntprobnp]
    }).sort_values("shap_value", key=np.abs, ascending=False)

    return sv, feature_names, contributions


# -----------------------------
# Stacked ensemble prediction (XGB + DL -> Meta)
# -----------------------------
def _build_attention_mlp(in_c: int, in_r: int, n_cls: int, drop: float = 0.3):
    class AttentionFusion(nn.Module):
        def __init__(self, dims):
            super().__init__()
            total = sum(dims)
            hidden = max(64, total // 2)
            self.net = nn.Sequential(
                nn.Linear(total, hidden), nn.ReLU(), nn.Linear(hidden, len(dims))
            )
        def forward(self, embs):
            concat = torch.cat(embs, dim=1)
            w = torch.softmax(self.net(concat), dim=1)
            fused = torch.cat([embs[i] * w[:, i].unsqueeze(1) for i in range(len(embs))], dim=1)
            return fused, w

    def mlp_layers(in_dim, sizes, drop):
        seq = []
        prev = in_dim
        for h in sizes:
            seq += [nn.Linear(prev, h), nn.InstanceNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        return nn.Sequential(*seq), prev

    class AttentionMLP(nn.Module):
        def __init__(self, in_c, in_r, n_cls, drop):
            super().__init__()
            self.clin, co = (mlp_layers(in_c, (64, 32), drop) if in_c > 0 else (None, 0))
            self.rad, ro = (mlp_layers(in_r, (256, 128, 64), drop) if in_r > 0 else (None, 0))
            dims = [d for d in (co, ro) if d > 0]
            self.fusion = AttentionFusion(dims)
            self.cls = nn.Sequential(nn.Linear(sum(dims), 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, n_cls))
        def forward(self, xc, xr):
            embs = []
            if self.clin: embs.append(self.clin(xc))
            if self.rad: embs.append(self.rad(xr))
            fused, w = self.fusion(embs)
            return self.cls(fused)

    return AttentionMLP(in_c, in_r, n_cls, drop)


def predict_with_stacked(age: float, lvef: float, troponin: float, ntprobnp: float):
    if not os.path.exists(STACKED_PATH) or not os.path.exists(CSV_PATH):
        raise FileNotFoundError("Stacked ensemble file or CSV not found")

    stacked_assets = joblib.load(STACKED_PATH)
    meta = stacked_assets.get("meta_learner")
    xgb_model = stacked_assets.get("xgb_model")
    label_encoder = stacked_assets.get("label_encoder")
    scalers = stacked_assets.get("scalers", {})
    sc_c = scalers.get("clinical", None)
    sc_r = scalers.get("radiomic", None)

    # For XGB scaling, use scaler from best_prognostic if available
    scaler_xgb = None
    if os.path.exists(BEST_XGB_PATH):
        try:
            best_assets = joblib.load(BEST_XGB_PATH)
            scaler_xgb = best_assets.get("scaler", None)
        except Exception:
            pass

    df_all = pd.read_csv(CSV_PATH)
    # Remove non-feature columns used in training
    drop_cols = [c for c in ["PatientID", "Risk_Score", "Risk_Category", "Reasoning"] if c in df_all.columns]
    features_df = df_all.drop(columns=drop_cols, errors='ignore')
    # Fill with column medians, and override clinical with user inputs
    medians = features_df.median(numeric_only=True)
    row = medians.copy()
    for col, val in zip(CLINICAL_COLS, [age, lvef, troponin, ntprobnp]):
        if col in features_df.columns:
            row[col] = float(val)
    X_full = row.to_frame().T

    # XGB probabilities
    if xgb_model is None:
        raise ValueError("xgb_model missing in stacked ensemble")
    X_xgb_input = X_full
    if scaler_xgb is not None:
        X_xgb = scaler_xgb.transform(X_xgb_input.values)
    else:
        X_xgb = X_xgb_input.values
    probs_xgb = xgb_model.predict_proba(X_xgb)[0]

    # DL probabilities via AttentionMLP if possible
    probs_dl = None
    if TORCH_AVAILABLE and sc_c is not None and sc_r is not None and "dl_model_state_dict" in stacked_assets:
        try:
            # Split features
            rad_cols = [c for c in features_df.columns if c.startswith("original_")]
            Xc = X_full[[c for c in CLINICAL_COLS if c in X_full.columns]].astype(float).values
            Xr = X_full[rad_cols].astype(float).values if len(rad_cols)>0 else np.zeros((1,0), dtype=float)
            Xc_s = sc_c.transform(Xc)
            Xr_s = sc_r.transform(Xr) if Xr.shape[1] > 0 else Xr
            n_cls = len(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else 4
            model_dl = _build_attention_mlp(Xc_s.shape[1], Xr_s.shape[1], n_cls)
            sd = stacked_assets["dl_model_state_dict"]
            model_dl.load_state_dict(sd, strict=False)
            model_dl.eval()
            with torch.no_grad():
                tens_c = torch.tensor(Xc_s).float()
                tens_r = torch.tensor(Xr_s).float()
                logits = model_dl(tens_c, tens_r)
                probs_dl = torch.softmax(logits, dim=1).cpu().numpy()[0]
        except Exception as e:
            _w.warn(f"DL branch failed, falling back to XGB-only in ensemble: {e}")

    # If DL not available, duplicate XGB probs as a proxy to satisfy meta input shape
    if probs_dl is None:
        probs_dl = probs_xgb.copy()

    X_meta = np.hstack([probs_xgb, probs_dl]).reshape(1, -1)
    probs_final = meta.predict_proba(X_meta)[0]
    pred_idx = int(np.argmax(probs_final))
    classes = list(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else ["Low Risk","Moderate Risk","High Risk (Chronic Heart Failure)","Very High Risk (Acute Cardiac Event)"]
    return classes[pred_idx], probs_final, pred_idx, classes


# -----------------------------
# MRI + U-Net Grad-CAM utilities
# -----------------------------
def load_nii_volume(path: str):
    itk_img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(itk_img)  # Z, Y, X
    vol = np.transpose(vol, (2, 1, 0))     # X, Y, Z
    return vol


def preprocess_volume_for_unet(vol: np.ndarray, target_size=(128, 128)):
    slices = []
    vol = vol.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    for i in range(vol.shape[2]):
        sl = vol[:, :, i]
        sl = tf.image.resize(sl[..., None], target_size, method="bilinear").numpy().squeeze()
        slices.append(sl)
    X = np.array(slices)[..., None]
    return X


def generate_gradcam_overlay(unet_model, vol: np.ndarray, target_class_idx: int = 2):
    X = preprocess_volume_for_unet(vol)
    preds = unet_model.predict(X, verbose=0)  # (n, h, w, C)
    masks = np.argmax(preds, axis=-1)

    # Choose slice with max myocardium pixels
    slice_scores = np.sum(masks == target_class_idx, axis=(1, 2))
    best_idx = int(np.argmax(slice_scores))
    proc_slice = X[best_idx:best_idx+1]  # (1, h, w, 1)
    orig_slice = vol[:, :, best_idx]

    # Build grad model using a stable conv layer target
    target_layer_name = None
    # Prefer explicit target layer if present
    preferred_names = [
        'c6_conv2_gradcam_target',  # defined in some training scripts
        'conv2d_5', 'conv2d_6'      # fallbacks (depend on build order)
    ]
    for nm in preferred_names:
        if nm in [l.name for l in unet_model.layers]:
            target_layer_name = nm
            break
    if target_layer_name is None:
        # Fallback to last Conv2D in the graph
        for layer in reversed(unet_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer_name = layer.name
                break
    if target_layer_name is None:
        raise RuntimeError("No Conv2D layer found for Grad-CAM")

    grad_model = tf.keras.models.Model(
        [unet_model.inputs],
        [unet_model.get_layer(target_layer_name).output, unet_model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(proc_slice)
        loss = tf.reduce_mean(predictions[:, :, :, target_class_idx])
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    fmap = conv_out[0].numpy()  # (h, w, channels)
    heatmap = np.tensordot(fmap, pooled_grads, axes=([2], [0]))
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # Resize heatmap back to original slice size
    heatmap_resized = tf.image.resize(heatmap[..., None], orig_slice.shape, method="bilinear").numpy().squeeze()

    # Overlay
    cmap = plt.get_cmap('jet')
    heat_rgba = (cmap(heatmap_resized) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert('RGBA')

    base = (255 * (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)).astype(np.uint8)
    base_img = Image.fromarray(base).convert('L').convert('RGBA')

    alpha = 0.35
    blended = Image.blend(base_img, heat_img, alpha)
    return blended, best_idx


def build_unet_multiclass(input_shape=(128, 128, 1), num_classes=3):
    inputs = layers.Input(input_shape, name='input_layer_2')
    # Encoder
    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c1_conv1')(inputs)
    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c1_conv2')(c1)
    p1 = layers.MaxPooling2D((2,2), name='p1')(c1)

    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c2_conv1')(p1)
    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c2_conv2')(c2)
    p2 = layers.MaxPooling2D((2,2), name='p2')(c2)

    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_conv1')(p2)
    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c3_conv2')(c3)
    p3 = layers.MaxPooling2D((2,2), name='p3')(c3)

    # Bottleneck
    bn = layers.Conv2D(128, (3,3), activation='relu', padding='same', name='bn_conv1')(p3)
    bn = layers.Conv2D(128, (3,3), activation='relu', padding='same', name='bn_conv2')(bn)

    # Decoder
    u1 = layers.UpSampling2D((2,2), name='u1')(bn)
    u1 = layers.concatenate([u1, c3], name='u1_concat')
    c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c4_conv1')(u1)
    c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='c4_conv2')(c4)

    u2 = layers.UpSampling2D((2,2), name='u2')(c4)
    u2 = layers.concatenate([u2, c2], name='u2_concat')
    c5 = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c5_conv1')(u2)
    c5 = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='c5_conv2')(c5)

    u3 = layers.UpSampling2D((2,2), name='u3')(c5)
    u3 = layers.concatenate([u3, c1], name='u3_concat')
    c6 = layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c6_conv1')(u3)
    c6 = layers.Conv2D(16, (3,3), activation='relu', padding='same', name='c6_conv2_gradcam_target')(c6)

    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax', name='final_output_layer')(c6)
    return models.Model(inputs, outputs)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Heart Risk Stratification", layout="wide")

# Theming tweaks
st.markdown(
    """
    <style>
      .main {background-color: #0b1220; color: #e6edf3;}
      section[data-testid="stSidebar"] {background-color: #111827;}
      .stButton>button {background-color: #2563EB; color: white; border-radius: 8px; padding: 0.6rem 1rem;}
      .stSelectbox>div>div>div {color: #e6edf3;}
      h1, h2, h3 {color: #60A5FA;}
      .metric-card {background: #111827; padding: 0.8rem 1rem; border-radius: 10px; border: 1px solid #1f2937;}
      .good {color:#10B981;} .warn {color:#F59E0B;} .bad {color:#EF4444;} .vh {color:#DC2626;}
      .info {color:#93C5FD;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Heart Risk Stratification — Clinical + MRI (Grad-CAM)")

with st.sidebar:
    st.header("Clinical Inputs")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
    lvef = st.number_input("LVEF (%)", min_value=0.0, max_value=80.0, value=55.0)
    troponin = st.number_input("Troponin (ng/L or µg/L)", min_value=0.0, max_value=50000.0, value=10.0)
    ntprobnp = st.number_input("NT-proBNP (pg/mL)", min_value=0.0, max_value=100000.0, value=300.0)
    uploaded = st.file_uploader("Upload 3D MRI (.nii.gz)", type=["nii.gz"]) 
    run_btn = st.button("Run Prediction")

tabs = st.tabs(["Results", "Explanations", "MRI Grad-CAM"]) 
res_col, shap_col, gradcam_col = tabs

if run_btn:
    # Rule-based reasoning (always available)
    rule_risk, rule_reason = classify_patient_risk_rule_based(age, lvef, troponin, ntprobnp)

    # Clinical model
    with st.spinner("Loading/Training clinical model..."):
        clinical_assets = load_or_train_clinical_model()
    model_pred_label, probs, pred_idx = predict_clinical_model(clinical_assets, age, lvef, troponin, ntprobnp)

    # Display classifications without plots; show dropdown with percentages
    with res_col:
        st.subheader("Risk Prediction 🔮")
        # Prefer stacked ensemble; fallback to clinical model
        try:
            model_pred_label, probs, pred_idx, classes = predict_with_stacked(age, lvef, troponin, ntprobnp)
            st.caption("Using stacked ensemble (XGB + DL → Meta)")
        except Exception as e_stack:
            st.caption("Using clinical model (fallback)")
            model_pred_label, probs, pred_idx = predict_clinical_model(clinical_assets, age, lvef, troponin, ntprobnp)
            classes = clinical_assets["classes"]

        options = [f"{cls} — {100.0*probs[i]:.1f}%" for i, cls in enumerate(classes)]
        selected = st.selectbox("Model probabilities (select to view)", options, index=pred_idx)

        # Styled metric card
        st.markdown("<div class='metric-card'>" +
                    f"<h3>Final Model Risk: <span class='info'>{model_pred_label}</span> ⚡</h3>" +
                    "</div>", unsafe_allow_html=True)

        st.subheader("Reasoning")
        st.markdown(f"- **Rule-based**: {rule_risk}")
        st.markdown(f"- **Rule explanation**: {rule_reason}")

    # SHAP explanation
    with shap_col:
        st.subheader("SHAP Explanation 🧠")
        def try_shap_from_best_model():
            # Attempt SHAP using best_prognostic_model.pkl and background from CSV
            assets_path = "best_prognostic_model.pkl"
            if not os.path.exists(assets_path) or not os.path.exists(CSV_PATH):
                raise FileNotFoundError("best_prognostic_model.pkl or CSV not found")
            best_assets = joblib.load(assets_path)
            model_best = best_assets.get("model", None)
            scaler_best = best_assets.get("scaler", None)
            label_map = best_assets.get("label_mapping", None)
            if model_best is None or label_map is None:
                raise ValueError("best_prognostic_model.pkl missing required keys")
            classes_best = [label_map[k] if isinstance(label_map.get(k, k), str) else k for k in sorted(label_map.keys())] if isinstance(label_map, dict) else clinical_assets["classes"]

            df_all = pd.read_csv(CSV_PATH)
            # Build a single-row input aligned to CSV columns
            X_row = df_all.iloc[:1].copy()
            # zero/NaN baseline, then set known clinicals
            X_row.iloc[0, :] = np.nan
            for col, val in zip(CLINICAL_COLS, [age, lvef, troponin, ntprobnp]):
                if col in X_row.columns:
                    X_row.loc[X_row.index[0], col] = float(val)

            # Background: sample valid rows from CSV
            bg = df_all.drop(columns=[c for c in ["PatientID", "Risk_Score", "Risk_Category", "Reasoning"] if c in df_all.columns], errors='ignore')
            bg = bg.sample(min(50, len(bg)), random_state=42)

            # Align to model input
            X_input = X_row.drop(columns=[c for c in ["PatientID", "Risk_Score", "Risk_Category", "Reasoning"] if c in X_row.columns], errors='ignore')

            # Scale if scaler exists; tolerate NaNs by filling with column means (from bg) for scaling only
            if scaler_best is not None:
                bg_filled = bg.fillna(bg.mean(numeric_only=True))
                bg_scaled = scaler_best.transform(bg_filled.values)
                x_filled = X_input.fillna(bg.mean(numeric_only=True))
                X_scaled = scaler_best.transform(x_filled.values)
                explainer = shap.Explainer(model_best.predict_proba, bg_scaled)
                exp = explainer(X_scaled)
            else:
                explainer = shap.Explainer(model_best.predict_proba, bg.values)
                exp = explainer(X_input.values)

            # Pred class index using model_best
            probs_best = model_best.predict_proba((X_scaled if scaler_best is not None else X_input.values))[0]
            pred_idx_best = int(np.nanargmax(probs_best))
            cls_names = classes_best if classes_best else clinical_assets["classes"]

            # Extract shap values for predicted class
            vals = np.array(exp.values)
            if vals.ndim == 3 and vals.shape[0] >= 1:
                sv = vals[0, :, pred_idx_best]
            elif vals.ndim == 2:
                sv = vals[0]
            else:
                sv = vals.squeeze()
            feat_names = (bg.columns.tolist())

            # Restrict display to known clinicals when available; else top features overall
            disp_idx = [feat_names.index(c) for c in CLINICAL_COLS if c in feat_names]
            if len(disp_idx) > 0:
                feat_names_disp = [feat_names[i] for i in disp_idx]
                sv_disp = sv[disp_idx]
                vals_disp = [age, lvef, troponin, ntprobnp][:len(disp_idx)]
            else:
                # take top-10 absolute
                order = np.argsort(np.abs(sv))[::-1][:10]
                feat_names_disp = [feat_names[i] for i in order]
                sv_disp = sv[order]
                vals_disp = [np.nan]*len(order)

            contrib_df = pd.DataFrame({
                "feature": feat_names_disp,
                "shap_value": sv_disp,
                "value": vals_disp
            }).sort_values("shap_value", key=np.abs, ascending=False)

            return sv_disp, feat_names_disp, contrib_df, probs_best, cls_names, pred_idx_best

        try:
            sv_use, fn_use, contrib_df, probs_best, cls_best, pred_idx_best = try_shap_from_best_model()
            # Show a compact bar of shap for display features
            fig, ax = plt.subplots(figsize=(6, 4))
            order = np.argsort(np.abs(sv_use))[::-1]
            ax.bar(np.array(fn_use)[order], sv_use[order], color=["#10B981" if v>0 else "#EF4444" for v in sv_use[order]])
            ax.set_ylabel("SHAP value")
            ax.set_title("Feature contributions (predicted class)")
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig)

            # Top positive and negative contributors table
            pos = contrib_df[contrib_df.shap_value > 0].head(3)
            neg = contrib_df[contrib_df.shap_value < 0].head(3)
            st.markdown("**Top positive contributors**")
            st.dataframe(pos.reset_index(drop=True))
            st.markdown("**Top negative contributors**")
            st.dataframe(neg.reset_index(drop=True))
        except Exception as e_best:
            st.info("Falling back to clinical-only SHAP due to compatibility.")
            try:
                shap_vals, feat_names, contrib_df = shap_explain_instance(clinical_assets, age, lvef, troponin, ntprobnp)
                fig, ax = plt.subplots(figsize=(6, 4))
                order = np.argsort(np.abs(shap_vals))[::-1]
                ax.bar(np.array(feat_names)[order], shap_vals[order], color=["#10B981" if v>0 else "#EF4444" for v in shap_vals[order]])
                ax.set_ylabel("SHAP value")
                ax.set_title("Feature contributions (predicted class)")
                plt.xticks(rotation=30, ha='right')
                st.pyplot(fig)

                pos = contrib_df[contrib_df.shap_value > 0].head(3)
                neg = contrib_df[contrib_df.shap_value < 0].head(3)
                st.markdown("**Top positive contributors**")
                st.dataframe(pos.reset_index(drop=True))
                st.markdown("**Top negative contributors**")
                st.dataframe(neg.reset_index(drop=True))
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

    # Grad-CAM overlay if MRI provided and U-Net available
    with gradcam_col:
        st.subheader("Grad-CAM Myocardium Heatmap (U-Net)")
        if uploaded is None:
            st.info("Upload a .nii.gz MRI to generate Grad-CAM overlay.")
        else:
            if not os.path.exists(UNET_MODEL_PATH):
                st.warning(f"U-Net model '{UNET_MODEL_PATH}' not found. Place the trained model file in the project root.")
            else:
                try:
                    with st.spinner("Loading MRI and running U-Net + Grad-CAM..."):
                        tmp_path = "_tmp_uploaded.nii.gz"
                        with open(tmp_path, 'wb') as f:
                            f.write(uploaded.read())
                        vol = load_nii_volume(tmp_path)
                        try:
                            unet_model = models.load_model(UNET_MODEL_PATH, compile=False)
                        except Exception:
                            unet_model = build_unet_multiclass(input_shape=(128,128,1), num_classes=3)
                            unet_model.load_weights(UNET_MODEL_PATH)
                        overlay_img, best_idx = generate_gradcam_overlay(unet_model, vol, target_class_idx=2)
                        st.image(overlay_img, caption=f"Grad-CAM overlay (best slice index {best_idx})", use_column_width=True)
                        os.remove(tmp_path)
                except Exception as e:
                    st.error(f"Grad-CAM generation failed: {e}")


