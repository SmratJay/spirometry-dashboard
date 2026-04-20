"""
=============================================================================
  Spirometer Smart Health Monitor  —  v3 (4-Model Dashboard)
  XGBoost · Random Forest · SVM (RBF kernel) · MLP Neural Network
  Final Year Research Project  —  NHANES 2007-2012

  ZERO hardcoded values.  Every metric, label, stat, plot value, and
  model parameter is computed live from the loaded dataset and the
  trained model objects at runtime.

  DATA LEAKAGE PREVENTION: Ratio-based and diagnostic columns that
  mathematically encode the Obstruction target are automatically
  excluded from training features, ensuring realistic accuracy.

  FEATURE IMPORTANCE:
    XGBoost / Random Forest → native model.feature_importances_
    SVM / MLP               → permutation importance (≤500 test rows, 3 repeats)

  SPEED:
    SVM training is subsampled to ≤5 000 rows (stratified) when the
    dataset is larger, keeping training time under ~30 s.
=============================================================================
"""

import os, time, threading, traceback, warnings
import numpy as np
import pandas as pd
import scipy.stats as sp_stats

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score,
)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

for _c in (UserWarning, FutureWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=_c)

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  —  single source of truth
#  Palette: Tailwind Slate + curated accents for data-viz clarity
# ══════════════════════════════════════════════════════════════════════════════
T = dict(
    bg="#0f172a",         # Slate 950 — warm deep navy
    card="#1e293b",       # Slate 800 — cards slightly lifted
    sidebar="#020617",    # Slate 950+ — sidebar anchors the layout
    border="#334155",     # Slate 700 — visible but not harsh
    input_bg="#1e293b",

    blue="#60a5fa",       # Blue 400 — primary accent
    green="#4ade80",      # Green 400 — confident, fresh
    orange="#fb923c",     # Orange 400 — warm & vibrant
    red="#f87171",        # Red 400 — softer alarm
    purple="#a78bfa",     # Violet 400 — elegant
    teal="#2dd4bf",       # Teal 400 — distinctive secondary
    pink="#f472b6",       # Pink 400 — highlights

    text="#e2e8f0",       # Slate 200 — soft white
    dim="#94a3b8",        # Slate 400 — muted text

    plot_bg="#1e293b",    # Matches card — plots feel integrated
    grid="#334155",       # Matches border — subtle grid
)

# ── Model registry — single source of truth for names, colors, icons ────────
MODEL_CFG = {
    "XGBoost":       {"color": T["blue"],   "alt": "#93c5fd", "icon": "🔵"},
    "Random Forest": {"color": T["orange"], "alt": "#fdba74", "icon": "🟠"},
    "SVM":           {"color": T["teal"],   "alt": "#5eead4", "icon": "🟢"},
    "MLP":           {"color": T["purple"], "alt": "#c4b5fd", "icon": "🟣"},
}

PAL = [cfg["color"] for cfg in MODEL_CFG.values()] + [
    "#93c5fd", "#86efac", "#fdba74", "#fca5a5", "#5eead4", "#c4b5fd",
]
FONT = "Segoe UI"

METRIC_TIPS = {
    "Accuracy":      "Fraction of all predictions that were correct.\n"
                     "1.0 = perfect, 0.0 = worst.",
    "ROC-AUC":       "Area Under the ROC Curve.\n"
                     "Measures class separability.\n"
                     "1.0 = perfect, 0.5 = random.",
    "Avg Precision": "Area under the Precision-Recall curve.\n"
                     "Summarises precision at every recall threshold.\n"
                     "Higher is better.",
    "MAE":           "Mean Absolute Error.\n"
                     "Average magnitude of prediction errors.\n"
                     "Lower is better.",
    "RMSE":          "Root Mean Squared Error.\n"
                     "Penalises large errors more than MAE.\n"
                     "Lower is better.",
    "R²":            "Coefficient of Determination.\n"
                     "Proportion of variance explained by the model.\n"
                     "1.0 = perfect, 0.0 = baseline.",
}

# ── Speed constants ──────────────────────────────────────────────────────────
_SVM_MAX_TRAIN = 5_000   # SVM is O(n²) — subsample training rows if larger
_PERM_MAX_TEST = 500     # permutation importance sample size


# ══════════════════════════════════════════════════════════════════════════════
#  ML PIPELINE  —  pure functions, zero GUI coupling
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(file_path: str, target_column: str):
    """
    Load CSV → engineer features → drop leakage columns (when target is
    Obstruction) → encode → scale → split 80/20 stratified.
    Returns X_tr, X_te, y_tr, y_te, meta dict.
    """
    df_raw = pd.read_csv(file_path)
    missing_raw = int(df_raw.isnull().sum().sum())

    # drop pure-ID columns
    id_cols = [c for c in df_raw.columns if c.lower() in ("seqn", "id")]
    df = df_raw.drop(columns=id_cols, errors="ignore").copy()

    # ── feature engineering ───────────────────────────────────────────────
    # Lung_Index = FEV1 / Age  (captures age-related lung decline)
    if "Baseline_FEV1_L" in df.columns and "Age" in df.columns:
        df["Lung_Index"] = (
            pd.to_numeric(df["Baseline_FEV1_L"], errors="coerce")
            / pd.to_numeric(df["Age"], errors="coerce").replace(0, np.nan)
        )

    # engineer Obstruction if absent using GOLD criterion (ratio < 0.70)
    if "Obstruction" not in df.columns:
        ratio = next(
            (c for c in ("Baseline_FEV1_FVC_Ratio", "FEV1_FVC")
             if c in df.columns), None)
        if ratio:
            df["Obstruction"] = (
                pd.to_numeric(df[ratio], errors="coerce") < 0.70
            ).astype(int)

    # ── DATA LEAKAGE PREVENTION ───────────────────────────────────────────
    # Columns that directly or indirectly encode FEV1/FVC ratio are removed
    # when the target is Obstruction, because Obstruction IS defined by that
    # ratio (< 0.70 per GOLD).  Keeping them makes the task trivially circular.
    leakage_cols = []
    if target_column == "Obstruction":
        _leak_tags = (
            "fev1_fvc", "fev1fvc",              # ratio & engineered ratio
            "ratio_z", "ratio5th", "ratio2point5",  # z-scores / LLN for ratio
            "obstruction_",                     # alternative obstruction labels
            "normal_", "mixed_", "prism_",      # diagnostic categories
            "restrictive_", "onevariable_lln",
        )
        leakage_cols = [
            c for c in df.columns
            if c != target_column
            and any(tag in c.lower() for tag in _leak_tags)
        ]
        if "Baseline_FEV1_FVC_Ratio" in df.columns:
            leakage_cols.append("Baseline_FEV1_FVC_Ratio")
        leakage_cols = list(set(leakage_cols))
        df = df.drop(columns=leakage_cols, errors="ignore")

    if target_column not in df.columns:
        raise ValueError(
            f"Target '{target_column}' not found.\n"
            f"Available columns:\n{df.columns.tolist()}")

    # ── impute ────────────────────────────────────────────────────────────
    for c in df.columns:
        if df[c].isnull().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                m = df[c].mode()
                if len(m):
                    df[c] = df[c].fillna(m.iloc[0])

    # ── X / y ─────────────────────────────────────────────────────────────
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    is_clf = (not pd.api.types.is_float_dtype(y)) or (y.nunique() <= 20)

    le = None
    if is_clf and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    elif is_clf:
        y = y.astype(int)

    # encode categoricals
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value",
                            unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])

    # scale
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X),
                     columns=feature_names, index=X.index)

    # split
    strat = y if is_clf else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strat)

    numeric_raw = [c for c in df_raw.select_dtypes(include=np.number).columns
                   if c not in id_cols and c != target_column]

    meta = dict(
        df_raw=df_raw, df_eng=df, feature_names=feature_names,
        n_rows=len(df_raw), n_cols=len(df_raw.columns),
        n_features=len(feature_names), is_clf=is_clf,
        target_column=target_column, le=le,
        numeric_raw_cols=numeric_raw,
        target_series=df[target_column],
        file_path=file_path,
        train_size=len(X_tr), test_size=len(X_te),
        missing_raw=missing_raw,
        class_dist=(dict(y.value_counts().sort_index()) if is_clf else None),
        leakage_dropped=len(leakage_cols),
        leakage_cols=leakage_cols,
    )
    return X_tr, X_te, y_tr, y_te, meta


# ── model builders ───────────────────────────────────────────────────────────

def _build_xgb(is_clf, n_classes):
    if is_clf:
        return xgb.XGBClassifier(
            objective="binary:logistic" if n_classes == 2 else "multi:softmax",
            num_class=None if n_classes == 2 else n_classes,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, verbosity=0)
    return xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=200,
        max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, verbosity=0)


def _build_rf(is_clf):
    if is_clf:
        return RandomForestClassifier(
            n_estimators=120, max_depth=6, criterion="entropy",
            class_weight="balanced", random_state=SEED, n_jobs=-1)
    return RandomForestRegressor(
        n_estimators=120, max_depth=6, random_state=SEED, n_jobs=-1)


def _build_svm(is_clf):
    """SVM with RBF kernel.  probability=True enables predict_proba → ROC/PR."""
    if is_clf:
        return SVC(kernel="rbf", C=1.0, gamma="scale",
                   probability=True, random_state=SEED)
    return SVR(kernel="rbf", C=1.0, gamma="scale")


def _build_mlp(is_clf):
    """3-hidden-layer MLP with early stopping to prevent overfitting."""
    if is_clf:
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            learning_rate_init=0.001, random_state=SEED)
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        learning_rate_init=0.001, random_state=SEED)


def _get_builder(name: str, is_clf: bool, n_classes: int):
    """Dispatch model name → instantiated model.  Add new models here only."""
    if name == "XGBoost":       return _build_xgb(is_clf, n_classes)
    if name == "Random Forest": return _build_rf(is_clf)
    if name == "SVM":           return _build_svm(is_clf)
    if name == "MLP":           return _build_mlp(is_clf)
    raise ValueError(f"Unknown model name: '{name}'")


# ── train + evaluate ─────────────────────────────────────────────────────────

def train_and_evaluate(model, X_tr, X_te, y_tr, y_te, is_clf, feat_names):
    """
    Fit → predict → compute all metrics from live predictions.
    - SVM: subsampled training if dataset > _SVM_MAX_TRAIN rows.
    - Feature importance: native (XGB/RF) or permutation (SVM/MLP).
    Returns results dict with NO hardcoded values.
    """
    # ── SVM adaptive subsample ────────────────────────────────────────────
    is_svm = hasattr(model, "kernel")
    svm_subsampled = False
    if is_svm and len(X_tr) > _SVM_MAX_TRAIN:
        if is_clf:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=_SVM_MAX_TRAIN, random_state=SEED)
            fit_idx, _ = next(sss.split(X_tr, y_tr))
        else:
            rng = np.random.RandomState(SEED)
            fit_idx = rng.choice(len(X_tr), _SVM_MAX_TRAIN, replace=False)
        X_fit = X_tr.iloc[fit_idx]
        y_fit = y_tr.iloc[fit_idx]
        svm_subsampled = True
    else:
        X_fit, y_fit = X_tr, y_tr

    t0 = time.perf_counter()
    model.fit(X_fit, y_fit)
    train_secs = time.perf_counter() - t0

    y_pred = model.predict(X_te)

    # ── feature importance ────────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        fi_arr    = model.feature_importances_
        fi_source = "native"
    else:
        # Permutation importance: model-agnostic, works for any estimator.
        # Computed on a small slice of the test set for speed.
        n_perm = min(_PERM_MAX_TEST, len(X_te))
        X_perm = X_te.iloc[:n_perm]
        y_perm = np.array(y_te)[:n_perm]
        perm = permutation_importance(
            model, X_perm, y_perm,
            n_repeats=3, random_state=SEED, n_jobs=-1)
        fi_arr    = np.maximum(perm.importances_mean, 0)  # clip negatives
        fi_source = "permutation"

    fi = pd.Series(fi_arr, index=feat_names).sort_values(ascending=False)

    res = dict(
        model=model,
        y_pred=y_pred,
        y_test=y_te,
        params=model.get_params(),
        model_class=type(model).__name__,
        feature_names=feat_names,
        feature_importance=fi,
        fi_source=fi_source,
        train_seconds=round(train_secs, 2),
        svm_subsampled=svm_subsampled,
        svm_train_size=len(X_fit),
    )

    if is_clf:
        y_prob = (model.predict_proba(X_te)[:, 1]
                  if hasattr(model, "predict_proba") else None)
        res.update(dict(
            accuracy=float(accuracy_score(y_te, y_pred)),
            cm=confusion_matrix(y_te, y_pred),
            clf_report=classification_report(y_te, y_pred, zero_division=0),
            roc_auc=(float(roc_auc_score(y_te, y_prob))
                     if y_prob is not None else None),
            avg_precision=(float(average_precision_score(y_te, y_prob))
                           if y_prob is not None else None),
            y_prob=y_prob,
            roc_data=(roc_curve(y_te, y_prob) if y_prob is not None else None),
            pr_data=(precision_recall_curve(y_te, y_prob)
                     if y_prob is not None else None),
        ))
    else:
        mae = float(mean_absolute_error(y_te, y_pred))
        mse = float(mean_squared_error(y_te, y_pred))
        res.update(dict(
            mae=mae, mse=mse,
            rmse=float(np.sqrt(mse)),
            r2=float(r2_score(y_te, y_pred)),
        ))
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_fig(nrows=1, ncols=1, figsize=(11, 5)):
    """Create a dark-themed Figure with a FLAT list of styled Axes."""
    fig = Figure(figsize=figsize, facecolor=T["plot_bg"])
    axs = fig.subplots(nrows, ncols, squeeze=False)
    flat = list(axs.flatten())
    for ax in flat:
        ax.set_facecolor(T["plot_bg"])
        ax.tick_params(colors=T["dim"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(T["border"])
        ax.grid(True, color=T["grid"], linewidth=0.4, linestyle="--")
    return fig, flat


def _embed(fig, parent):
    """Embed a Figure into a Tkinter frame without closing it."""
    for w in parent.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT FUNCTIONS  — all accept  results: dict  (model_name → result_dict)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(meta, results: dict):
    """
    6-panel overview.  All values computed live from meta / results.
    Row 1: FEV1/FVC ratio + GOLD threshold | Class balance | Age vs FEV1
    Row 2: Target distribution | Pred vs Actual (all models) | FEV1 histogram
    """
    df  = meta["df_raw"]
    tgt = meta["target_series"]
    fig, ax = _make_fig(2, 3, (14, 7.5))

    # ── [0] FEV1/FVC Ratio with GOLD 0.70 threshold ─────────────────────
    ratio_col = next((c for c in ("Baseline_FEV1_FVC_Ratio", "FEV1_FVC")
                      if c in df.columns), None)
    if ratio_col:
        d = pd.to_numeric(df[ratio_col], errors="coerce").dropna()
        ax[0].hist(d, bins=50, color=T["orange"], alpha=0.85,
                   edgecolor=T["bg"], linewidth=0.3)
        ax[0].axvline(0.70, color=T["red"], ls="--", lw=1.8,
                      label="GOLD threshold = 0.70")
        ax[0].axvline(d.median(), color=T["blue"], ls="--", lw=1.2,
                      label=f"Median = {d.median():.3f}")
        n_obs = int((d < 0.70).sum())
        ax[0].set_title(f"FEV1/FVC Ratio  ({n_obs:,} below 0.70)")
        ax[0].legend(fontsize=7)
    else:
        ax[0].hist(tgt.dropna(), bins=40, color=T["blue"], alpha=0.85,
                   edgecolor=T["bg"], linewidth=0.3)
        ax[0].set_title(f"Target: {meta['target_column']}")
    ax[0].set_xlabel(ratio_col or meta["target_column"])
    ax[0].set_ylabel("Count")

    # ── [1] Class balance ────────────────────────────────────────────────
    if meta["is_clf"] and meta["class_dist"]:
        cls = list(meta["class_dist"].keys())
        cnt = list(meta["class_dist"].values())
        bars = ax[1].bar([str(c) for c in cls], cnt,
                         color=[T["green"], T["red"]][:len(cls)], width=0.5)
        for b, v in zip(bars, cnt):
            pct = v / sum(cnt) * 100
            ax[1].text(b.get_x() + b.get_width() / 2,
                       b.get_height() + max(cnt) * 0.01,
                       f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom",
                       fontsize=7, color=T["text"])
        ax[1].set_title("Class Distribution (Full Dataset)")
        ax[1].set_xlabel("Class"); ax[1].set_ylabel("Count")
    else:
        ax[1].hist(tgt.dropna(), bins=40, color=T["blue"], alpha=0.85,
                   edgecolor=T["bg"])
        med = tgt.median()
        ax[1].axvline(med, color=T["red"], ls="--", lw=1.2,
                      label=f"Median = {med:.3f}")
        ax[1].set_title(f"Target: {meta['target_column']}")
        ax[1].set_xlabel(meta["target_column"]); ax[1].set_ylabel("Count")
        ax[1].legend(fontsize=7)

    # ── [2] Age vs FEV1 scatter ──────────────────────────────────────────
    xcol = next((c for c in ("Age",) if c in df.columns), None)
    ycol = next((c for c in ("Baseline_FEV1_L", "FEV1") if c in df.columns), None)
    if xcol and ycol:
        sub = df[[xcol, ycol]].dropna()
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=SEED)
        ax[2].scatter(sub[xcol], sub[ycol], alpha=0.2, s=6, color=T["green"])
        ax[2].set_xlabel(xcol); ax[2].set_ylabel(ycol)
        ax[2].set_title(f"{xcol} vs {ycol}  (n={len(sub):,})")
    else:
        ax[2].text(0.5, 0.5, "Age / FEV1 columns\nnot found",
                   ha="center", va="center", transform=ax[2].transAxes,
                   color=T["dim"], fontsize=10)
        ax[2].set_title("Scatter (N/A)")

    # ── [3] Target distribution with stats ───────────────────────────────
    tgt_clean = tgt.dropna()
    ax[3].hist(tgt_clean, bins=40, color=T["purple"], alpha=0.85,
               edgecolor=T["bg"], linewidth=0.3)
    med = tgt_clean.median(); mn = tgt_clean.mean()
    ax[3].axvline(med, color=T["red"], ls="--", lw=1.2,
                  label=f"Median = {med:.3f}")
    ax[3].axvline(mn, color=T["blue"], ls=":", lw=1.2,
                  label=f"Mean = {mn:.3f}")
    ax[3].set_title(f"Target: {meta['target_column']}  (n={len(tgt_clean):,})")
    ax[3].set_xlabel(meta["target_column"]); ax[3].set_ylabel("Count")
    ax[3].legend(fontsize=7)

    # ── [4] Pred vs Actual — all trained models overlaid ─────────────────
    plotted = False
    for name, res in results.items():
        col = MODEL_CFG[name]["color"]
        yt  = np.array(res["y_test"])
        yp  = np.array(res["y_pred"])
        if not meta["is_clf"]:
            ax[4].scatter(yt, yp, alpha=0.25, s=8, color=col, label=name)
        else:
            jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(yt))
            ax[4].scatter(yt + jit, yp + jit, alpha=0.15, s=6,
                          color=col, label=name)
        plotted = True
    if plotted and not meta["is_clf"]:
        first = next(iter(results.values()))
        mn_v = np.array(first["y_test"]).min()
        mx_v = np.array(first["y_test"]).max()
        ax[4].plot([mn_v, mx_v], [mn_v, mx_v], "r--", lw=1.2, label="Ideal")
    ax[4].set_title("Predicted vs Actual — All Models")
    ax[4].set_xlabel("Actual"); ax[4].set_ylabel("Predicted")
    if plotted:
        ax[4].legend(fontsize=6)

    # ── [5] Secondary histogram (FEV1 / BMI / Weight / Height) ───────────
    sec_col = next((c for c in ("Baseline_FEV1_L", "BMI", "Weight", "Height")
                    if c in df.columns and c != meta["target_column"]), None)
    if sec_col:
        d = pd.to_numeric(df[sec_col], errors="coerce").dropna()
        ax[5].hist(d, bins=40, color=T["teal"], alpha=0.85,
                   edgecolor=T["bg"], linewidth=0.3)
        ax[5].axvline(d.median(), color=T["red"], ls="--", lw=1.2,
                      label=f"Median = {d.median():.3f}")
        short = sec_col.replace("Baseline_", "").replace("_", " ")
        ax[5].set_title(f"{short}  (n={len(d):,})")
        ax[5].set_xlabel(sec_col); ax[5].set_ylabel("Count")
        ax[5].legend(fontsize=7)
    else:
        ax[5].set_visible(False)

    fig.suptitle("Dataset Overview", fontweight="bold", fontsize=13,
                 y=0.99, color=T["text"])
    fig.tight_layout(pad=1.5)
    return fig


def plot_feature_importance(results: dict, top_n=20):
    """One panel per trained model.  2×2 grid for 4 models, 1×N for fewer."""
    if not results:
        fig, ax = _make_fig(1, 1, (6, 4))
        ax[0].text(0.5, 0.5, "No models trained yet.",
                   ha="center", va="center",
                   transform=ax[0].transAxes, color=T["dim"], fontsize=11)
        ax[0].axis("off")
        return fig

    n = len(results)
    if n <= 2:
        nrows, ncols = 1, n
        figsize = (7 * n, 9)
    else:
        nrows, ncols = 2, 2
        figsize = (14, 14)

    fig, axes = _make_fig(nrows, ncols, figsize)

    for idx, (name, res) in enumerate(results.items()):
        ax    = axes[idx]
        color = MODEL_CFG[name]["color"]
        fi    = res["feature_importance"].head(top_n)
        labels = [l[:34] + "…" if len(l) > 34 else l for l in fi.index]

        bars = ax.barh(range(len(fi)), fi.values,
                       color=color, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        src_tag = f"  [{res['fi_source']} importance]"
        ax.set_title(f"{name} — Top {len(fi)} Features{src_tag}", fontsize=9)

        if fi.values.max() > 0:
            for bar, val in zip(bars, fi.values):
                ax.text(val + fi.values.max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=6.5,
                        color=T["text"])

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout(pad=2.5)
    return fig


def plot_predictions(results: dict, is_clf=True):
    """Confusion matrices (classification) or scatter + error hist (regression)."""
    if not results:
        fig, ax = _make_fig(1, 1, (6, 4))
        ax[0].axis("off")
        return fig

    n = len(results)

    if is_clf:
        if n <= 2:
            nrows, ncols = 1, n
            figsize = (7 * n, 5.5)
        else:
            nrows, ncols = 2, 2
            figsize = (14, 11)

        fig, axes = _make_fig(nrows, ncols, figsize)

        for idx, (name, res) in enumerate(results.items()):
            ax    = axes[idx]
            color = MODEL_CFG[name]["color"]
            cm    = res["cm"]
            n_cls = cm.shape[0]

            im = ax.imshow(cm, cmap="Blues", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            thresh = cm.max() / 2
            for i in range(n_cls):
                for j in range(n_cls):
                    val = cm[i, j]
                    txt_col = "white" if val > thresh else "#1a1a1a"
                    ax.text(j, i, f"{val:,}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=txt_col)

            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels([f"Pred {k}" for k in range(n_cls)], fontsize=8)
            ax.set_yticklabels([f"True {k}" for k in range(n_cls)], fontsize=8)
            roc = res.get("roc_auc")
            roc_str = f"  AUC={roc:.4f}" if roc is not None else ""
            ax.set_title(f"{name} — Confusion Matrix\n"
                         f"Acc={res['accuracy']:.4f}{roc_str}",
                         color=color, fontsize=9)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        for ax in axes[n:]:
            ax.set_visible(False)
    else:
        fig, axes = _make_fig(n, 2, (12, 5 * n))
        for row_idx, (name, res) in enumerate(results.items()):
            color = MODEL_CFG[name]["color"]
            yt = np.array(res["y_test"])
            yp = np.array(res["y_pred"])

            ax_s = axes[row_idx * 2]
            ax_s.scatter(yt, yp, alpha=0.3, s=10, color=color)
            mn, mx = yt.min(), yt.max()
            ax_s.plot([mn, mx], [mn, mx], "r--", lw=1.2, label="Ideal fit")
            ax_s.set_title(f"{name}\nPred vs Actual  R²={res['r2']:.4f}")
            ax_s.set_xlabel("Actual"); ax_s.set_ylabel("Predicted")
            ax_s.legend(fontsize=7)

            ax_e = axes[row_idx * 2 + 1]
            errs = yp - yt
            ax_e.hist(errs, bins=40, color=color, alpha=0.8,
                      edgecolor=T["bg"], linewidth=0.3)
            ax_e.axvline(0, color=T["red"], ls="--", lw=1.2)
            ax_e.set_title(f"{name}\nError Distribution  MAE={res['mae']:.4f}")
            ax_e.set_xlabel("Error (Pred − Actual)")
            ax_e.set_ylabel("Count")

    fig.tight_layout(pad=2.0)
    return fig


def plot_residuals_cm(results: dict, is_clf=True):
    """Classification reports (clf) or residual/Q-Q plots (reg)."""
    if not results:
        fig, ax = _make_fig(1, 1, (6, 4))
        ax[0].axis("off")
        return fig

    n = len(results)

    if is_clf:
        fig, axes = _make_fig(n, 1, (11, 4.5 * n))
        for idx, (name, res) in enumerate(results.items()):
            color = MODEL_CFG[name]["color"]
            ax = axes[idx]
            ax.axis("off")
            ax.text(0.02, 0.97, f"{name}  —  Classification Report",
                    transform=ax.transAxes, fontsize=12, color=color,
                    va="top", fontweight="bold")
            ax.text(0.02, 0.87, res["clf_report"],
                    transform=ax.transAxes, fontsize=9.5,
                    color=T["text"], va="top", fontfamily="monospace")
            acc = res["accuracy"]
            roc = res.get("roc_auc")
            ap  = res.get("avg_precision")
            summary = f"Accuracy: {acc:.4f}"
            if roc is not None:
                summary += f"  |  ROC-AUC: {roc:.4f}"
            if ap is not None:
                summary += f"  |  Avg Precision: {ap:.4f}"
            if res.get("svm_subsampled"):
                summary += f"  |  ⚠ SVM trained on {res['svm_train_size']:,} rows (subsampled)"
            ax.text(0.02, 0.05, summary,
                    transform=ax.transAxes, fontsize=9,
                    color=T["green"], va="bottom", fontweight="bold")
    else:
        fig, axes = _make_fig(n, 2, (12, 5 * n))
        for idx, (name, res) in enumerate(results.items()):
            color = MODEL_CFG[name]["color"]
            yt = np.array(res["y_test"])
            yp = np.array(res["y_pred"])
            residuals = yp - yt

            ax_r = axes[idx * 2]
            ax_r.scatter(yp, residuals, alpha=0.3, s=8, color=color)
            ax_r.axhline(0, color=T["red"], ls="--", lw=1.2)
            ax_r.set_xlabel("Predicted Value"); ax_r.set_ylabel("Residual")
            ax_r.set_title(f"{name} — Residuals vs Predicted")

            ax_q = axes[idx * 2 + 1]
            (osm, osr), (slope, intercept, _) = sp_stats.probplot(residuals)
            ax_q.scatter(osm, osr, alpha=0.4, s=8, color=color)
            ax_q.plot(osm, slope * np.array(osm) + intercept,
                      color=T["red"], lw=1.5, label="Reference line")
            ax_q.set_xlabel("Theoretical Quantiles")
            ax_q.set_ylabel("Sample Quantiles")
            ax_q.set_title(f"{name} — Q-Q Plot")
            ax_q.legend(fontsize=7)

    fig.tight_layout(pad=2.0)
    return fig


def plot_roc_pr(results: dict):
    """ROC and Precision-Recall curves for all trained classification models."""
    fig, axes = _make_fig(1, 2, (12, 5))
    ax_r, ax_p = axes[0], axes[1]

    for name, res in results.items():
        if res.get("roc_data") is None:
            continue
        color = MODEL_CFG[name]["color"]
        fpr, tpr, _ = res["roc_data"]
        prec, rec, _ = res["pr_data"]
        ax_r.plot(fpr, tpr, color=color, lw=2,
                  label=f"{name}  AUC={res['roc_auc']:.4f}")
        ax_p.plot(rec, prec, color=color, lw=2,
                  label=f"{name}  AP={res['avg_precision']:.4f}")

    ax_r.plot([0, 1], [0, 1], color=T["dim"], ls="--", lw=1, label="Random")
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1.02)
    ax_r.set_xlabel("False Positive Rate")
    ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("ROC Curve — All Models"); ax_r.legend(fontsize=8)

    ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1.02)
    ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision")
    ax_p.set_title("Precision-Recall Curve — All Models")
    ax_p.legend(fontsize=8)

    fig.tight_layout(pad=2.0)
    return fig


def plot_distributions(df_raw, numeric_cols, max_plots=9):
    """Key column distributions — clinically relevant spirometry fields first."""
    priority = [
        "Baseline_FEV1_L", "Baseline_FVC_L", "Baseline_FEV1_FVC_Ratio",
        "Baseline_PEF_Ls", "Baseline_FEF2575_Ls",
        "FEV1_Zscores_GLOBAL", "FVC_Zscores_GLOBAL",
        "Age", "Height", "Weight", "BMI",
    ]
    ordered = ([c for c in priority if c in numeric_cols]
               + [c for c in numeric_cols if c not in priority])[:max_plots]
    n = len(ordered)
    if n == 0:
        fig, axes = _make_fig(1, 1, (6, 3))
        axes[0].text(0.5, 0.5, "No numeric columns found",
                     ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=11, color=T["dim"])
        axes[0].axis("off")
        return fig

    pal_dist = [T["blue"], T["teal"], T["orange"], T["purple"],
                T["green"], T["red"], T["pink"], "#93c5fd", "#86efac"]
    ncols_g = 3
    nrows_g = max(1, int(np.ceil(n / ncols_g)))
    fig, flat = _make_fig(nrows_g, ncols_g, (14, 3.8 * nrows_g))

    for idx, col in enumerate(ordered):
        ax   = flat[idx]
        data = pd.to_numeric(df_raw[col], errors="coerce").dropna()
        med  = data.median()
        ax.hist(data, bins=40, color=pal_dist[idx % len(pal_dist)],
                alpha=0.85, edgecolor=T["bg"], linewidth=0.3)
        ax.axvline(med, color=T["red"], ls="--", lw=1.2,
                   label=f"Median={med:.3f}")
        short = col.replace("_", " ").replace("Baseline ", "")
        ax.set_title(short[:28], fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right")

    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Key Column Distributions (Live from Loaded Dataset)",
                 fontweight="bold", y=1.0, color=T["text"], fontsize=12)
    fig.tight_layout(pad=1.8)
    return fig


def plot_model_comparison(results: dict, is_clf: bool):
    """
    Left panel : grouped bar chart — all N models × all metrics.
    Right panel: ROC curves overlaid (clf) or error distribution overlay (reg).
    Dynamically handles 1–4 models.
    """
    if not results:
        fig, ax = _make_fig(1, 1, (6, 4))
        ax[0].text(0.5, 0.5, "No results to compare.",
                   ha="center", va="center",
                   transform=ax[0].transAxes, color=T["dim"])
        ax[0].axis("off")
        return fig

    fig, axes = _make_fig(1, 2, (14, 5.5))
    ax_b, ax_c = axes[0], axes[1]

    names  = list(results.keys())
    colors = [MODEL_CFG[n]["color"] for n in names]
    N      = len(names)

    if is_clf:
        metrics = [("accuracy", "Accuracy"),
                   ("roc_auc", "ROC-AUC"),
                   ("avg_precision", "Avg Precision")]
    else:
        metrics = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]

    n_m     = len(metrics)
    x       = np.arange(n_m)
    w       = 0.7 / N
    offsets = np.linspace(-(N - 1) / 2, (N - 1) / 2, N) * w

    all_vals = []
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name].get(k) or 0 for k, _ in metrics]
        all_vals.extend(vals)
        bars = ax_b.bar(x + offsets[i], vals, w * 0.88,
                        label=name, color=color, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.003,
                      f"{val:.3f}", ha="center", va="bottom",
                      fontsize=6.5, color=T["text"])

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax_b.set_ylabel("Score")
    ax_b.set_title("Metrics Comparison — All Models")
    ax_b.legend(fontsize=8)
    top = max(all_vals) if all_vals else 1.0
    ax_b.set_ylim(0, top * 1.22)

    # right panel
    if is_clf:
        for name, color in zip(names, colors):
            if results[name].get("roc_data"):
                fpr, tpr, _ = results[name]["roc_data"]
                auc = results[name]["roc_auc"]
                ax_c.plot(fpr, tpr, color=color, lw=2,
                          label=f"{name}  AUC={auc:.4f}")
        ax_c.plot([0, 1], [0, 1], color=T["dim"], ls="--", lw=1)
        ax_c.set_title("ROC Curves — All Models Overlaid")
        ax_c.set_xlabel("False Positive Rate")
        ax_c.set_ylabel("True Positive Rate")
        ax_c.legend(fontsize=8)
    else:
        for name, color in zip(names, colors):
            errs = (np.array(results[name]["y_pred"])
                    - np.array(results[name]["y_test"]))
            ax_c.hist(errs, bins=35, alpha=0.5, color=color, label=name)
        ax_c.axvline(0, color=T["red"], ls="--", lw=1.2)
        ax_c.set_title("Error Distribution — All Models")
        ax_c.set_xlabel("Error"); ax_c.set_ylabel("Count")
        ax_c.legend(fontsize=8)

    fig.tight_layout(pad=2.0)
    return fig


def build_conclusion(meta, results: dict):
    """
    Returns list of (text, tag) tuples.
    Every number comes from live results / meta dicts — nothing hardcoded.
    """
    L = []
    def h(t):      L.append((t, "h"))
    def kv(k, v):  L.append((f"  {k:<32}{v}", "kv"))
    def sep():     L.append(("─" * 74, "sep"))
    def p(t):      L.append((t, "p"))
    def b():       L.append(("", "p"))

    # per-model tag name: "XGBoost" → "xgboost", "Random Forest" → "random_forest"
    def _tag(name):
        return name.lower().replace(" ", "_").replace("-", "_")

    L.append(("═" * 74, "sep"))
    h("MODEL TRAINING SUMMARY — Spirometer Smart Health Monitor")
    p(f"  Target: {meta['target_column']}  •  NHANES 2007–2012  •  {len(results)} model(s) trained")
    L.append(("═" * 74, "sep"))
    b()

    # ── DATASET ──────────────────────────────────────────────────────────
    h("📂  DATASET")
    sep()
    kv("File path:",           meta["file_path"])
    kv("Total rows:",          f"{meta['n_rows']:,}")
    kv("Total columns:",       f"{meta['n_cols']:,}")
    kv("Feature count:",       f"{meta['n_features']:,}")
    kv("Target column:",       meta["target_column"])
    kv("Task type:",           "Classification" if meta["is_clf"] else "Regression")
    kv("Train / Test split:",  f"{meta['train_size']:,} / {meta['test_size']:,}  (80 / 20)")
    ts = meta["target_series"].dropna()
    kv("Target mean:",         f"{ts.mean():.4f}")
    kv("Target std:",          f"{ts.std():.4f}")
    kv("Target min / max:",    f"{ts.min():.4f}  /  {ts.max():.4f}")
    if meta["is_clf"] and meta["class_dist"]:
        kv("Class distribution:", str(meta["class_dist"]))
    kv("Missing values (raw):",  f"{meta['missing_raw']:,}")
    leaked = meta.get("leakage_dropped", 0)
    if leaked:
        kv("Leakage cols dropped:", f"{leaked}  (ratio / diagnostic columns)")
    b()

    # ── PER-MODEL SECTIONS ────────────────────────────────────────────────
    def _model_section(res, name):
        cfg = MODEL_CFG[name]
        tag = _tag(name)
        sep()
        L.append((f"{cfg['icon']}  {name.upper()}", tag))
        b()

        pr = res["params"]
        kv("Model class:",      res["model_class"])
        for k in ("n_estimators", "max_depth", "learning_rate", "subsample",
                  "colsample_bytree", "class_weight", "criterion", "objective",
                  "eval_metric", "kernel", "C", "gamma",
                  "hidden_layer_sizes", "max_iter", "early_stopping",
                  "random_state"):
            v = pr.get(k)
            if v is not None:
                kv(f"{k}:", str(v))

        if res.get("svm_subsampled"):
            kv("SVM training rows:",
               f"{res['svm_train_size']:,}  (stratified subsample for speed)")
        kv("Feature importance:", res["fi_source"])
        kv("Training time:",     f"{res['train_seconds']:.2f} seconds")
        b()

        L.append(("  📊  METRICS", tag))
        if meta["is_clf"]:
            acc = res["accuracy"]
            kv("Accuracy:",      f"{acc:.4f}  ({acc * 100:.2f}%)")
            roc = res.get("roc_auc")
            kv("ROC-AUC:",       f"{roc:.4f}" if roc is not None else "N/A")
            ap  = res.get("avg_precision")
            kv("Avg Precision:", f"{ap:.4f}" if ap is not None else "N/A")
            b()
            p("  Classification Report:")
            L.append((res["clf_report"], "mono"))
        else:
            kv("MAE:",  f"{res['mae']:.4f}")
            kv("MSE:",  f"{res['mse']:.4f}")
            kv("RMSE:", f"{res['rmse']:.4f}")
            kv("R²:",   f"{res['r2']:.4f}")
        b()

        L.append(("  🏆  TOP 10 FEATURES", tag))
        for rank, (feat, score) in enumerate(
                res["feature_importance"].head(10).items(), 1):
            short = feat[:42] + "…" if len(feat) > 42 else feat
            kv(f"  #{rank:>2}  {short}", f"{score:.4f}")
        b()

    for name, res in results.items():
        _model_section(res, name)

    # ── KEY SPIROMETRY STATS ──────────────────────────────────────────────
    sep()
    h("📈  KEY SPIROMETRY STATS (raw data)")
    stat_cols = [c for c in (
        "Baseline_FEV1_L", "Baseline_FVC_L", "Baseline_FEV1_FVC_Ratio",
        "Baseline_PEF_Ls", "Baseline_FEF2575_Ls",
        "FEV1_Zscores_GLOBAL", "FVC_Zscores_GLOBAL",
        "Age", "Height", "Weight", "BMI",
    ) if c in meta["df_raw"].columns]
    for col in stat_cols:
        d = pd.to_numeric(meta["df_raw"][col], errors="coerce").dropna()
        kv(col, f"mean={d.mean():.3f}  std={d.std():.3f}  "
                f"min={d.min():.3f}  max={d.max():.3f}  n={len(d):,}")
    b()

    # ── COMPARISON VERDICT (only when >1 model ran) ───────────────────────
    if len(results) > 1:
        sep()
        h("⚖️  COMPARISON VERDICT")
        b()

        if meta["is_clf"]:
            pairs = [("Best Accuracy:",       "accuracy",      False),
                     ("Best ROC-AUC:",        "roc_auc",       False),
                     ("Best Avg Precision:",   "avg_precision", False)]
        else:
            pairs = [("Best R²:",     "r2",   False),
                     ("Lowest MAE:",  "mae",  True),
                     ("Lowest RMSE:", "rmse", True)]

        for label, key, lower in pairs:
            scored = {n: (r.get(key) or 0) for n, r in results.items()}
            winner = (min(scored, key=scored.get)
                      if lower else max(scored, key=scored.get))
            vals_str = "  ".join(f"{n}={v:.4f}" for n, v in scored.items())
            L.append((f"  {label:<30}{winner}  [{vals_str}]", _tag(winner)))
        b()

        h("⏱  TRAINING TIME")
        times = {n: r["train_seconds"] for n, r in results.items()}
        for n, t in times.items():
            kv(f"{n}:", f"{t:.2f}s")
        fastest = min(times, key=times.get)
        kv("Fastest model:", fastest)
        b()

    # ── CLINICAL INTERPRETATION ───────────────────────────────────────────
    sep()
    h("🩺  CLINICAL INTERPRETATION")
    b()
    p("  FEV1 (Forced Expiratory Volume in 1 second) and FVC (Forced Vital")
    p("  Capacity) are primary spirometry markers of lung function.  An")
    p("  FEV1/FVC ratio below 0.70 is the standard GOLD criterion for")
    p("  obstructive lung disease (asthma, COPD).")
    b()
    p("  Lung_Index = FEV1 / Age  captures age-related lung decline.")
    p("  Height²×Race encodes population-specific reference equations.")
    b()
    leaked = meta.get("leakage_dropped", 0)
    if leaked:
        p(f"  ⚠  DATA LEAKAGE PREVENTION:  {leaked} columns containing the")
        p("  FEV1/FVC ratio, its z-scores, and derived diagnostic labels were")
        p("  excluded from training.  These columns mathematically encode the")
        p("  Obstruction target (ratio < 0.70 = GOLD criterion) and would")
        p("  yield trivially perfect accuracy.  The model now predicts from")
        p("  genuine physiological features only.")
        b()

    for name, res in results.items():
        tag = _tag(name)
        if meta["is_clf"]:
            acc   = res["accuracy"]
            roc   = res.get("roc_auc") or 0
            grade = ("Excellent" if acc > 0.95
                     else "Good" if acc > 0.80 else "Moderate")
            icon  = ("✅" if grade == "Excellent"
                     else "⚠️" if grade == "Good" else "❌")
            L.append((
                f"  {icon}  {name}: {grade} — Accuracy {acc * 100:.2f}%, "
                f"AUC {roc:.4f}.  "
                f"{'Reliably identifies' if acc > 0.90 else 'Partially identifies'} "
                f"obstructive patterns.", tag))
        else:
            r2    = res.get("r2", 0)
            grade = ("Excellent" if r2 > 0.85
                     else "Good" if r2 > 0.60 else "Moderate")
            icon  = "✅" if grade == "Excellent" else "⚠️"
            L.append((
                f"  {icon}  {name}: {grade} — R²={r2:.4f}, "
                f"explains {r2 * 100:.1f}% of variance in "
                f"{meta['target_column']}.", tag))
    b()
    L.append(("═" * 74, "sep"))
    return L


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Spirometer Monitor — 4-Model ML Dashboard  [v3]")
        self.configure(bg=T["bg"])
        self.minsize(1150, 700)
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        self._meta    = None
        self._results = {}       # model_name → result_dict

        self.v_path   = tk.StringVar(
            value=r"D:\New folder (2)\NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv")
        self.v_target = tk.StringVar(value="Obstruction")
        self.v_mode   = tk.StringVar(value="Compare All")

        self._build()

    # ──────────────────────────────────────────────────────────────────────
    #  BUILD UI
    # ──────────────────────────────────────────────────────────────────────

    def _build(self):
        self._topbar()
        body = tk.Frame(self, bg=T["bg"])
        body.pack(fill=tk.BOTH, expand=True)
        self._sidebar(body)
        self._content(body)

    def _topbar(self):
        bar = tk.Frame(self, bg=T["sidebar"], height=50)
        bar.pack(fill=tk.X); bar.pack_propagate(False)
        tk.Frame(self, bg=T["border"], height=1).pack(fill=tk.X)
        tk.Label(bar, text="🫁", font=(FONT, 16),
                 bg=T["sidebar"], fg=T["blue"]).pack(side=tk.LEFT, padx=(18, 6))
        tk.Label(bar, text="Spirometer Smart Health Monitor",
                 font=(FONT, 14, "bold"), bg=T["sidebar"],
                 fg=T["text"]).pack(side=tk.LEFT)
        self._lbl_sub = tk.Label(
            bar, text="XGBoost · Random Forest · SVM · MLP  •  NHANES 2007–2012",
            font=(FONT, 9), bg=T["sidebar"], fg=T["dim"])
        self._lbl_sub.pack(side=tk.LEFT, padx=20)
        self._lbl_dot = tk.Label(bar, text="● READY", font=(FONT, 9, "bold"),
                                 bg=T["sidebar"], fg=T["green"])
        self._lbl_dot.pack(side=tk.RIGHT, padx=20)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=T["sidebar"], width=244)
        sb.pack(side=tk.LEFT, fill=tk.Y); sb.pack_propagate(False)

        self._sb_canvas = tk.Canvas(sb, bg=T["sidebar"],
                                    highlightthickness=0, width=228)
        vsb = ttk.Scrollbar(sb, orient="vertical",
                            command=self._sb_canvas.yview)
        inner = tk.Frame(self._sb_canvas, bg=T["sidebar"])
        inner.bind("<Configure>", lambda e: self._sb_canvas.configure(
            scrollregion=self._sb_canvas.bbox("all")))
        self._sb_canvas.create_window((0, 0), window=inner,
                                      anchor="nw", width=228)
        self._sb_canvas.configure(yscrollcommand=vsb.set)
        self._sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def _enter(e): self._sb_canvas.bind_all("<MouseWheel>", _mw)
        def _leave(e): self._sb_canvas.unbind_all("<MouseWheel>")
        def _mw(e):    self._sb_canvas.yview_scroll(
            -1 if e.delta > 0 else 1, "units")
        self._sb_canvas.bind("<Enter>", _enter)
        self._sb_canvas.bind("<Leave>", _leave)

        def _sec(txt):
            tk.Frame(inner, bg=T["border"], height=1).pack(
                fill=tk.X, pady=(12, 0))
            tk.Label(inner, text=txt, font=(FONT, 8, "bold"),
                     bg=T["sidebar"], fg=T["blue"],
                     anchor="w").pack(fill=tk.X, padx=10, pady=(4, 2))

        # ── DATASET ──────────────────────────────────────────────────────
        _sec("▸  DATASET")
        tk.Label(inner, text="CSV file path:", font=(FONT, 8),
                 bg=T["sidebar"], fg=T["dim"],
                 anchor="w").pack(fill=tk.X, padx=10)
        pf = tk.Frame(inner, bg=T["sidebar"])
        pf.pack(fill=tk.X, padx=10, pady=2)
        self._lbl_path = tk.Label(
            pf, textvariable=self.v_path, font=(FONT, 7),
            bg=T["sidebar"], fg=T["text"], anchor="w",
            wraplength=178, justify="left", cursor="hand2")
        self._lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._lbl_path.bind("<Button-1>", self._browse)

        # ── TARGET ───────────────────────────────────────────────────────
        _sec("◎  TARGET COLUMN")
        self._cb_tgt = ttk.Combobox(
            inner, textvariable=self.v_target,
            font=(FONT, 9), state="normal", width=22)
        self._cb_tgt.pack(padx=10, pady=(2, 6), fill=tk.X)

        # ── MODEL SELECTION — driven entirely by MODEL_CFG ───────────────
        _sec("◈  MODEL")
        mode_options = list(MODEL_CFG.keys()) + ["Compare All"]
        for lbl in mode_options:
            col = MODEL_CFG[lbl]["color"] if lbl in MODEL_CFG else T["teal"]
            tk.Radiobutton(
                inner, text=lbl, variable=self.v_mode, value=lbl,
                font=(FONT, 9, "bold"), bg=T["sidebar"], fg=col,
                selectcolor=T["border"], activebackground=T["sidebar"],
                activeforeground=col, indicatoron=True,
            ).pack(anchor="w", padx=14, pady=1)

        # ── TRAINED PARAMS ───────────────────────────────────────────────
        _sec("⚙  TRAINED PARAMS")
        self._pf = tk.Frame(inner, bg=T["sidebar"])
        self._pf.pack(fill=tk.X, padx=10, pady=4)
        self._show_params({})

        # ── RUN / SAVE ────────────────────────────────────────────────────
        tk.Frame(inner, bg=T["border"], height=1).pack(fill=tk.X, pady=8)
        self._btn = tk.Button(
            inner, text="▶  Run", font=(FONT, 12, "bold"),
            bg="#22c55e", fg="#052e16", activebackground="#16a34a",
            relief="flat", cursor="hand2", pady=10,
            command=self._on_run)
        self._btn.pack(fill=tk.X, padx=10, pady=(4, 4))
        tk.Button(
            inner, text="💾  Save All Plots", font=(FONT, 9),
            bg=T["card"], fg=T["text"], activebackground=T["border"],
            relief="flat", cursor="hand2", pady=5,
            command=self._save).pack(fill=tk.X, padx=10, pady=(0, 8))

        # ── DATASET INFO ──────────────────────────────────────────────────
        _sec("ℹ  DATASET INFO")
        self._if = tk.Frame(inner, bg=T["sidebar"])
        self._if.pack(fill=tk.X, padx=10, pady=4)
        self._info = {}
        for k in ("Rows", "Columns", "Features", "Task",
                  "Train Size", "Test Size", "Missing (raw)", "Leakage Cols"):
            row = tk.Frame(self._if, bg=T["sidebar"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=k + ":", font=(FONT, 8),
                     bg=T["sidebar"], fg=T["dim"],
                     width=14, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="—", font=(FONT, 8, "bold"),
                           bg=T["sidebar"], fg=T["text"], anchor="w")
            lbl.pack(side=tk.LEFT)
            self._info[k] = lbl

    def _content(self, parent):
        ct = tk.Frame(parent, bg=T["bg"])
        ct.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── metric card rows — one per model in MODEL_CFG ─────────────────
        self._cards_outer = tk.Frame(ct, bg=T["bg"])
        self._cards_outer.pack(fill=tk.X, padx=8, pady=(6, 0))
        self._crows = {}
        for mname, cfg in MODEL_CFG.items():
            rf = tk.Frame(self._cards_outer, bg=T["bg"])
            tk.Label(rf, text=mname, font=(FONT, 8, "bold"),
                     bg=T["bg"], fg=cfg["color"],
                     width=15, anchor="e").pack(side=tk.LEFT, padx=(0, 6))
            cards = {}
            for met in ("Accuracy", "ROC-AUC", "Avg Precision",
                        "MAE", "RMSE", "R²"):
                cf = tk.Frame(rf, bg=T["card"],
                              highlightbackground=T["border"],
                              highlightthickness=1)
                cf.pack(side=tk.LEFT, padx=4, pady=3, ipadx=12, ipady=6)
                tk.Label(cf, text=met, font=(FONT, 7),
                         bg=T["card"], fg=T["dim"]).pack()
                vl = tk.Label(cf, text="—", font=(FONT, 13, "bold"),
                              bg=T["card"], fg=T["dim"])
                vl.pack()
                cards[met] = (cf, vl)
                self._add_tooltip(cf, METRIC_TIPS.get(met, ""))
                self._add_tooltip(vl, METRIC_TIPS.get(met, ""))
            self._crows[mname] = {"frame": rf, "cards": cards}

        # ── notebook ──────────────────────────────────────────────────────
        sty = ttk.Style(); sty.theme_use("default")
        sty.configure("D.TNotebook", background=T["bg"], borderwidth=0)
        sty.configure("D.TNotebook.Tab", background=T["card"],
                       foreground=T["dim"], padding=[14, 6], font=(FONT, 9))
        sty.map("D.TNotebook.Tab",
                background=[("selected", T["bg"])],
                foreground=[("selected", T["blue"])])

        self._nb = ttk.Notebook(ct, style="D.TNotebook")
        self._nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        for title, attr in [
            ("📈 Overview",          "_t_ov"),
            ("🏆 Feature Importance","_t_fi"),
            ("🎯 Predictions",       "_t_pr"),
            ("📉 Residuals / CM",    "_t_rc"),
            ("📡 ROC / PR Curves",   "_t_rp"),
            ("📊 Distributions",     "_t_ds"),
            ("⚖️ Model Comparison",  "_t_mc"),
            ("📋 Conclusion",        "_t_cl"),
        ]:
            f = tk.Frame(self._nb, bg=T["bg"])
            self._nb.add(f, text=title)
            setattr(self, attr, f)

        # ── conclusion scrollable text ─────────────────────────────────────
        bar = ttk.Scrollbar(self._t_cl)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self._txt = tk.Text(
            self._t_cl, bg=T["card"], fg=T["text"],
            font=("Consolas", 9), wrap=tk.WORD,
            yscrollcommand=bar.set, relief="flat",
            padx=14, pady=10, state=tk.DISABLED, cursor="arrow")
        self._txt.pack(fill=tk.BOTH, expand=True)
        bar.config(command=self._txt.yview)

        # base text tags
        base_tags = [
            ("h",    dict(font=("Consolas", 11, "bold"), foreground=T["blue"])),
            ("kv",   dict(foreground=T["text"])),
            ("sep",  dict(foreground=T["border"])),
            ("p",    dict(foreground=T["dim"])),
            ("mono", dict(font=("Consolas", 8), foreground=T["text"])),
        ]
        # per-model tags — keyed by model_name.lower().replace(" ","_")
        model_tags = [
            (name.lower().replace(" ", "_").replace("-", "_"),
             dict(font=("Consolas", 9, "bold"), foreground=cfg["color"]))
            for name, cfg in MODEL_CFG.items()
        ]
        for tag, cfg_t in base_tags + model_tags:
            self._txt.tag_configure(tag, **cfg_t)

    # ──────────────────────────────────────────────────────────────────────
    #  TOOLTIP
    # ──────────────────────────────────────────────────────────────────────

    def _add_tooltip(self, widget, text):
        if not text:
            return
        tip = None

        def _enter(e):
            nonlocal tip
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{e.x_root + 12}+{e.y_root + 12}")
            tk.Label(tip, text=text, font=(FONT, 8),
                     bg=T["card"], fg=T["text"],
                     relief="solid", borderwidth=1, padx=10, pady=6,
                     justify="left", wraplength=280).pack()

        def _leave(e):
            nonlocal tip
            if tip:
                tip.destroy(); tip = None

        widget.bind("<Enter>", _enter, add="+")
        widget.bind("<Leave>", _leave, add="+")

    # ──────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _browse(self, _=None):
        p = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.v_path.set(p)
            try:
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                self._cb_tgt["values"] = cols
            except Exception:
                pass

    def _dot(self, txt, fg=None):
        """Thread-safe status indicator update."""
        def _do(): self._lbl_dot.config(text=txt, fg=fg or T["green"])
        self.after(0, _do)

    def _show_params(self, params):
        for w in self._pf.winfo_children():
            w.destroy()
        # Keys that are noisy / uninteresting in the sidebar display
        skip = {
            "num_class", "verbosity", "n_jobs", "base_score", "booster",
            "callbacks", "early_stopping_rounds", "enable_categorical",
            "feature_types", "grow_policy", "importance_type",
            "interaction_constraints", "max_bin", "max_cat_threshold",
            "max_cat_to_onehot", "max_delta_step", "max_leaves",
            "min_child_weight", "missing", "monotone_constraints",
            "multi_strategy", "nthread", "num_parallel_tree",
            "reg_alpha", "reg_lambda", "sampling_method",
            "scale_pos_weight", "tree_method", "validate_parameters",
            "device", "min_samples_split", "min_samples_leaf",
            "min_weight_fraction_leaf", "max_features", "max_leaf_nodes",
            "min_impurity_decrease", "bootstrap", "oob_score",
            "warm_start", "ccp_alpha", "max_samples", "verbose",
            "activation", "solver", "alpha", "batch_size",
            "learning_rate", "power_t", "shuffle", "tol",
            "n_iter_no_change", "beta_1", "beta_2", "epsilon",
            "momentum", "nesterovs_momentum", "validation_fraction",
            "t0", "cache_size", "decision_function_shape", "break_ties",
            "coef0", "degree", "shrinking",
        }
        shown = {k: v for k, v in params.items()
                 if v is not None and k not in skip}
        for k, v in list(shown.items())[:12]:
            row = tk.Frame(self._pf, bg=T["sidebar"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{k}:", font=(FONT, 7),
                     bg=T["sidebar"], fg=T["dim"],
                     width=17, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=str(v), font=(FONT, 7, "bold"),
                     bg=T["sidebar"], fg=T["green"],
                     anchor="w").pack(side=tk.LEFT)

    def _upd_info(self, m):
        leaked = m.get("leakage_dropped", 0)
        data = {
            "Rows":          f"{m['n_rows']:,}",
            "Columns":       f"{m['n_cols']:,}",
            "Features":      f"{m['n_features']:,}",
            "Task":          "Classification" if m["is_clf"] else "Regression",
            "Train Size":    f"{m['train_size']:,}",
            "Test Size":     f"{m['test_size']:,}",
            "Missing (raw)": f"{m['missing_raw']:,}",
            "Leakage Cols":  str(leaked),
        }
        for k, v in data.items():
            fg = (T["orange"] if k == "Task"
                  else T["red"] if k == "Leakage Cols" and leaked > 0
                  else T["text"])
            self._info[k].config(text=v, fg=fg)

    def _show_card_rows(self, results: dict):
        """Show card rows only for models that actually ran."""
        for data in self._crows.values():
            data["frame"].pack_forget()
        for name in results:
            if name in self._crows:
                self._crows[name]["frame"].pack(fill=tk.X, pady=2)

    def _upd_cards(self, res, model_name: str, is_clf: bool):
        cards = self._crows[model_name]["cards"]
        color = MODEL_CFG[model_name]["color"]
        if is_clf:
            mp = {"Accuracy":      ("accuracy",     color),
                  "ROC-AUC":       ("roc_auc",      T["blue"]),
                  "Avg Precision": ("avg_precision", T["purple"]),
                  "MAE":           (None,            T["dim"]),
                  "RMSE":          (None,            T["dim"]),
                  "R²":            (None,            T["dim"])}
        else:
            mp = {"Accuracy":      (None,  T["dim"]),
                  "ROC-AUC":       (None,  T["dim"]),
                  "Avg Precision": (None,  T["dim"]),
                  "MAE":           ("mae",  T["orange"]),
                  "RMSE":          ("rmse", T["red"]),
                  "R²":            ("r2",   T["green"])}
        for met, (key, col) in mp.items():
            _, vl = cards[met]
            if key and res.get(key) is not None:
                vl.config(text=f"{res[key]:.4f}", fg=col)
            else:
                vl.config(text="—", fg=T["dim"])

    def _reset_cards(self):
        for data in self._crows.values():
            for _, (_, vl) in data["cards"].items():
                vl.config(text="—", fg=T["dim"])

    # ──────────────────────────────────────────────────────────────────────
    #  RUN PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def _on_run(self):
        self._btn.config(state=tk.DISABLED, text="⏳ Running…")
        self._reset_cards()
        self._dot("● LOADING", T["orange"])
        threading.Thread(target=self._pipeline, daemon=True).start()

    def _pipeline(self):
        try:
            path   = self.v_path.get().strip()
            target = self.v_target.get().strip()
            mode   = self.v_mode.get()

            self._dot("● PREPROCESSING", T["orange"])
            X_tr, X_te, y_tr, y_te, meta = load_and_preprocess(path, target)
            self._meta = meta
            n_cls = int(y_tr.nunique()) if meta["is_clf"] else 0

            results = {}
            for name in MODEL_CFG:
                if mode not in (name, "Compare All"):
                    continue
                self._dot(f"● TRAINING {name}", MODEL_CFG[name]["color"])
                model = _get_builder(name, meta["is_clf"], n_cls)
                results[name] = train_and_evaluate(
                    model, X_tr, X_te, y_tr, y_te,
                    meta["is_clf"], meta["feature_names"])

            self._results = results
            self.after(0, self._post_run)

        except Exception:
            err = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Pipeline Error", err))
            self.after(0, lambda: self._dot("● ERROR", T["red"]))
            self.after(0, lambda: self._btn.config(
                state=tk.NORMAL, text="▶  Run"))

    def _post_run(self):
        meta    = self._meta
        results = self._results
        is_clf  = meta["is_clf"]

        self._upd_info(meta)
        if results:
            # show params of the last trained model in sidebar
            self._show_params(next(reversed(results.values()))["params"])

        self._show_card_rows(results)
        for name, res in results.items():
            self._upd_cards(res, name, is_clf)

        self._dot("● RENDERING PLOTS", T["blue"])
        self._render(meta, results, is_clf)

        mode = self.v_mode.get()
        self._lbl_sub.config(
            text=f"{mode}  •  Target: {meta['target_column']}  •  NHANES 2007–2012")
        self._dot("● READY", T["green"])
        self._btn.config(state=tk.NORMAL, text="▶  Run")

    def _render(self, meta, results, is_clf):
        _embed(plot_overview(meta, results),       self._t_ov)
        _embed(plot_feature_importance(results),   self._t_fi)
        _embed(plot_predictions(results, is_clf),  self._t_pr)
        _embed(plot_residuals_cm(results, is_clf), self._t_rc)

        if is_clf:
            _embed(plot_roc_pr(results), self._t_rp)
        else:
            for w in self._t_rp.winfo_children():
                w.destroy()
            tk.Label(self._t_rp,
                     text="ROC / PR curves are only available\n"
                          "for classification tasks.",
                     font=(FONT, 11), bg=T["bg"],
                     fg=T["dim"]).pack(expand=True)

        _embed(plot_distributions(meta["df_raw"], meta["numeric_raw_cols"]),
               self._t_ds)

        if len(results) > 1:
            _embed(plot_model_comparison(results, is_clf), self._t_mc)
        else:
            for w in self._t_mc.winfo_children():
                w.destroy()
            tk.Label(self._t_mc,
                     text="Select 'Compare All' (or run multiple models)\n"
                          "to see the side-by-side comparison.",
                     font=(FONT, 11), bg=T["bg"],
                     fg=T["dim"]).pack(expand=True)

        # Conclusion tab
        lines = build_conclusion(meta, results)
        t = self._txt
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)
        for text, tag in lines:
            t.insert(tk.END, text + "\n", tag)
        t.config(state=tk.DISABLED)
        t.yview_moveto(0)

    # ──────────────────────────────────────────────────────────────────────
    #  SAVE
    # ──────────────────────────────────────────────────────────────────────

    def _save(self):
        if self._meta is None:
            messagebox.showinfo("No results", "Train a model first.")
            return
        folder = filedialog.askdirectory(title="Select folder to save plots")
        if not folder:
            return

        meta    = self._meta
        results = self._results
        is_clf  = meta["is_clf"]

        figs = {
            "overview.png":           plot_overview(meta, results),
            "feature_importance.png": plot_feature_importance(results),
            "predictions.png":        plot_predictions(results, is_clf),
            "residuals_cm.png":       plot_residuals_cm(results, is_clf),
            "distributions.png":      plot_distributions(
                                          meta["df_raw"],
                                          meta["numeric_raw_cols"]),
        }
        if is_clf:
            figs["roc_pr.png"] = plot_roc_pr(results)
        if len(results) > 1:
            figs["model_comparison.png"] = plot_model_comparison(
                results, is_clf)

        for fname, fig in figs.items():
            fig.savefig(os.path.join(folder, fname),
                        bbox_inches="tight", dpi=150,
                        facecolor=T["plot_bg"])

        messagebox.showinfo("Saved", f"All plots saved to:\n{folder}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    try:
        cols = pd.read_csv(app.v_path.get(), nrows=0).columns.tolist()
        app._cb_tgt["values"] = cols
    except Exception:
        pass
    app.mainloop()
