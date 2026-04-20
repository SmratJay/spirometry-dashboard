"""
=============================================================================
  Spirometer Smart Health Monitor  —  v4 (Modern CustomTkinter UI)
  XGBoost · Random Forest · SVM (RBF kernel) · MLP Neural Network
  Final Year Research Project  —  NHANES 2007-2012

  UI:      CustomTkinter (Bootstrap-quality dark theme, rounded widgets)
  PLOTS:   Matplotlib Figure() embedded via FigureCanvasTkAgg
  MODELS:  XGBoost, Random Forest, SVM, MLP — all live from dataset
  METRICS: Classification → Accuracy / ROC-AUC / Avg Precision only
           (MAE/RMSE/R² shown only for regression targets)
  LEAKAGE: 32 ratio/diagnostic columns auto-dropped for Obstruction target
  SPEED:   SVM subsampled to ≤5,000 training rows; MLP early stopping
=============================================================================
"""

import os, time, threading, traceback, warnings, re
import numpy as np
import pandas as pd
import scipy.stats as sp_stats

import matplotlib
matplotlib.use("TkAgg")
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
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

for _c in (UserWarning, FutureWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=_c)

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
#  THEME — CustomTkinter dark palette
# ══════════════════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Fine-grained color tokens (used for matplotlib + accent widgets)
C = dict(
    bg="#0b0f1a",          # near-black with blue tint — app background
    panel="#111827",       # sidebar / panel background
    card="#1a2235",        # card / elevated surface
    card2="#1e2a3a",       # hover / secondary card
    border="#2d3f55",      # subtle border
    accent="#3b82f6",      # Tailwind Blue-500 — primary CTA
    accent2="#1d4ed8",     # pressed state
    green="#10b981",       # Emerald-500
    orange="#f59e0b",      # Amber-500
    red="#ef4444",         # Red-500
    purple="#8b5cf6",      # Violet-500
    teal="#14b8a6",        # Teal-500
    pink="#ec4899",        # Pink-500
    text="#f1f5f9",        # Slate-100
    dim="#94a3b8",         # Slate-400
    plot_bg="#111827",
    grid="#1e2d3e",
)

MODEL_CFG = {
    "XGBoost":       {"color": C["accent"], "alt": "#93c5fd", "icon": "⬡", "ctk_color": "#3b82f6"},
    "Random Forest": {"color": C["orange"], "alt": "#fdba74", "icon": "⬡", "ctk_color": "#f59e0b"},
    "SVM":           {"color": C["teal"],   "alt": "#5eead4", "icon": "⬡", "ctk_color": "#14b8a6"},
    "MLP":           {"color": C["purple"], "alt": "#c4b5fd", "icon": "⬡", "ctk_color": "#8b5cf6"},
}

FONT_FAMILY = "Segoe UI"
PAL = [C["accent"], C["orange"], C["teal"], C["purple"],
       C["green"], C["red"], C["pink"], "#93c5fd", "#86efac", "#fdba74"]

_SVM_MAX_TRAIN = 5_000
_PERM_MAX_TEST = 500


def _grid_for(n_items: int):
    if n_items <= 1:
        return 1, 1
    if n_items == 2:
        return 1, 2
    return int(np.ceil(n_items / 2.0)), 2


def _downsample_xy(x, y, max_points=2500, seed=SEED):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  ML PIPELINE  (unchanged from v3 — pure functions, zero GUI coupling)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(file_path: str, target_column: str):
    df_raw = pd.read_csv(file_path)
    missing_raw = int(df_raw.isnull().sum().sum())
    id_cols = [c for c in df_raw.columns if c.lower() in ("seqn", "id")]
    df = df_raw.drop(columns=id_cols, errors="ignore").copy()

    if "Baseline_FEV1_L" in df.columns and "Age" in df.columns:
        df["Lung_Index"] = (
            pd.to_numeric(df["Baseline_FEV1_L"], errors="coerce")
            / pd.to_numeric(df["Age"], errors="coerce").replace(0, np.nan))

    if "Obstruction" not in df.columns:
        ratio = next((c for c in ("Baseline_FEV1_FVC_Ratio", "FEV1_FVC")
                      if c in df.columns), None)
        if ratio:
            df["Obstruction"] = (
                pd.to_numeric(df[ratio], errors="coerce") < 0.70).astype(int)

    leakage_cols = []
    if target_column == "Obstruction":
        _tags = ("fev1_fvc","fev1fvc","ratio_z","ratio5th","ratio2point5",
                 "obstruction_","normal_","mixed_","prism_",
                 "restrictive_","onevariable_lln")
        leakage_cols = [c for c in df.columns
                        if c != target_column
                        and any(t in c.lower() for t in _tags)]
        if "Baseline_FEV1_FVC_Ratio" in df.columns:
            leakage_cols.append("Baseline_FEV1_FVC_Ratio")
        leakage_cols = list(set(leakage_cols))
        df = df.drop(columns=leakage_cols, errors="ignore")

    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found.\n"
                         f"Columns: {df.columns.tolist()}")

    for c in df.columns:
        if df[c].isnull().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                m = df[c].mode()
                if len(m): df[c] = df[c].fillna(m.iloc[0])

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    is_clf = (not pd.api.types.is_float_dtype(y)) or (y.nunique() <= 20)

    le = None
    if is_clf and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    elif is_clf:
        y = y.astype(int)

    feature_defaults = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            feature_defaults[c] = float(pd.to_numeric(X[c], errors="coerce").median())
        else:
            m = X[c].mode()
            feature_defaults[c] = (m.iloc[0] if len(m) else "Unknown")

    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    oe = None
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=feature_names, index=X.index)

    strat = y if is_clf else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strat)

    numeric_raw = [c for c in df_raw.select_dtypes(include=np.number).columns
                   if c not in id_cols and c != target_column]

    return X_tr, X_te, y_tr, y_te, dict(
        df_raw=df_raw, df_eng=df, feature_names=feature_names,
        n_rows=len(df_raw), n_cols=len(df_raw.columns),
        n_features=len(feature_names), is_clf=is_clf,
        target_column=target_column, le=le,
        feature_defaults=feature_defaults,
        cat_cols=cat_cols,
        encoder=oe,
        scaler=scaler,
        numeric_raw_cols=numeric_raw, target_series=df[target_column],
        file_path=file_path, train_size=len(X_tr), test_size=len(X_te),
        missing_raw=missing_raw,
        class_dist=(dict(y.value_counts().sort_index()) if is_clf else None),
        leakage_dropped=len(leakage_cols), leakage_cols=leakage_cols,
    )


def _build_xgb(is_clf, n_classes):
    if is_clf:
        return xgb.XGBClassifier(
            objective="binary:logistic" if n_classes == 2 else "multi:softmax",
            num_class=None if n_classes == 2 else n_classes,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, verbosity=0)
    return xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=0)

def _build_rf(is_clf):
    if is_clf:
        return RandomForestClassifier(
            n_estimators=120, max_depth=6, criterion="entropy",
            class_weight="balanced", random_state=SEED, n_jobs=-1)
    return RandomForestRegressor(n_estimators=120, max_depth=6,
                                  random_state=SEED, n_jobs=-1)

def _build_svm(is_clf):
    if is_clf:
        return SVC(kernel="rbf", C=1.0, gamma="scale",
                   probability=True, random_state=SEED)
    return SVR(kernel="rbf", C=1.0, gamma="scale")

def _build_mlp(is_clf):
    if is_clf:
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                             early_stopping=True, validation_fraction=0.1,
                             learning_rate_init=0.001, random_state=SEED)
    return MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                        early_stopping=True, validation_fraction=0.1,
                        learning_rate_init=0.001, random_state=SEED)

def _get_builder(name, is_clf, n_classes):
    return {"XGBoost": _build_xgb(is_clf, n_classes),
            "Random Forest": _build_rf(is_clf),
            "SVM": _build_svm(is_clf),
            "MLP": _build_mlp(is_clf)}[name]


def train_and_evaluate(model, X_tr, X_te, y_tr, y_te, is_clf, feat_names):
    is_svm = hasattr(model, "kernel")
    svm_sub = False
    if is_svm and len(X_tr) > _SVM_MAX_TRAIN:
        if is_clf:
            sss = StratifiedShuffleSplit(1, train_size=_SVM_MAX_TRAIN,
                                         random_state=SEED)
            idx, _ = next(sss.split(X_tr, y_tr))
        else:
            idx = np.random.RandomState(SEED).choice(
                len(X_tr), _SVM_MAX_TRAIN, replace=False)
        X_fit, y_fit = X_tr.iloc[idx], y_tr.iloc[idx]
        svm_sub = True
    else:
        X_fit, y_fit = X_tr, y_tr

    t0 = time.perf_counter()
    model.fit(X_fit, y_fit)
    secs = time.perf_counter() - t0
    y_pred = model.predict(X_te)

    if hasattr(model, "feature_importances_"):
        fi_arr, fi_src = model.feature_importances_, "native"
    else:
        n = min(_PERM_MAX_TEST, len(X_te))
        pi = permutation_importance(model, X_te.iloc[:n], np.array(y_te)[:n],
                                     n_repeats=3, random_state=SEED, n_jobs=-1)
        fi_arr, fi_src = np.maximum(pi.importances_mean, 0), "permutation"

    fi = pd.Series(fi_arr, index=feat_names).sort_values(ascending=False)
    res = dict(model=model, y_pred=y_pred, y_test=y_te,
               params=model.get_params(), model_class=type(model).__name__,
               feature_names=feat_names, feature_importance=fi,
               fi_source=fi_src, train_seconds=round(secs, 2),
               svm_subsampled=svm_sub, svm_train_size=len(X_fit))

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
        res.update(mae=mae, mse=mse, rmse=float(np.sqrt(mse)),
                   r2=float(r2_score(y_te, y_pred)))
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_fig(nrows=1, ncols=1, figsize=(11, 5)):
    fig = Figure(figsize=figsize, facecolor=C["plot_bg"])
    axs = fig.subplots(nrows, ncols, squeeze=False)
    flat = list(axs.flatten())
    for ax in flat:
        ax.set_facecolor(C["plot_bg"])
        ax.tick_params(colors=C["dim"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.grid(True, color=C["grid"], linewidth=0.4, linestyle="--")
    return fig, flat


def _embed(fig, parent):
    for w in parent.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT FUNCTIONS  (accept results: dict — identical logic to v3)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(meta, results: dict):
    df, tgt = meta["df_raw"], meta["target_series"]
    fig, ax = _make_fig(2, 3, (14, 7.5))

    ratio_col = next((c for c in ("Baseline_FEV1_FVC_Ratio", "FEV1_FVC")
                      if c in df.columns), None)
    if ratio_col:
        d = pd.to_numeric(df[ratio_col], errors="coerce").dropna()
        ax[0].hist(d, bins=50, color=C["orange"], alpha=0.85,
                   edgecolor=C["bg"], linewidth=0.3)
        ax[0].axvline(0.70, color=C["red"], ls="--", lw=1.8,
                      label="GOLD threshold = 0.70")
        ax[0].axvline(d.median(), color=C["accent"], ls="--", lw=1.2,
                      label=f"Median = {d.median():.3f}")
        ax[0].set_title(f"FEV1/FVC Ratio  ({int((d<0.70).sum()):,} below 0.70)",
                        color=C["text"])
        ax[0].legend(fontsize=7)
    else:
        ax[0].hist(tgt.dropna(), bins=40, color=C["accent"], alpha=0.85,
                   edgecolor=C["bg"], linewidth=0.3)
        ax[0].set_title(f"Target: {meta['target_column']}", color=C["text"])
    ax[0].set_xlabel(ratio_col or meta["target_column"], color=C["dim"])
    ax[0].set_ylabel("Count", color=C["dim"])

    if meta["is_clf"] and meta["class_dist"]:
        cls = list(meta["class_dist"].keys())
        cnt = list(meta["class_dist"].values())
        bars = ax[1].bar([str(c) for c in cls], cnt,
                         color=[C["green"], C["red"]][:len(cls)], width=0.5)
        for b, v in zip(bars, cnt):
            pct = v / sum(cnt) * 100
            ax[1].text(b.get_x() + b.get_width() / 2,
                       b.get_height() + max(cnt) * 0.01,
                       f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom",
                       fontsize=7, color=C["text"])
        ax[1].set_title("Class Distribution", color=C["text"])
        ax[1].set_xlabel("Class", color=C["dim"])
        ax[1].set_ylabel("Count", color=C["dim"])
    else:
        ax[1].hist(tgt.dropna(), bins=40, color=C["accent"], alpha=0.85)
        med = tgt.median()
        ax[1].axvline(med, color=C["red"], ls="--", lw=1.2,
                      label=f"Median={med:.3f}")
        ax[1].set_title(f"Target: {meta['target_column']}", color=C["text"])
        ax[1].legend(fontsize=7)

    xcol = next((c for c in ("Age",) if c in df.columns), None)
    ycol = next((c for c in ("Baseline_FEV1_L", "FEV1") if c in df.columns), None)
    if xcol and ycol:
        sub = df[[xcol, ycol]].dropna()
        if len(sub) > 2000: sub = sub.sample(2000, random_state=SEED)
        ax[2].scatter(sub[xcol], sub[ycol], alpha=0.2, s=6, color=C["green"])
        ax[2].set_xlabel(xcol, color=C["dim"]); ax[2].set_ylabel(ycol, color=C["dim"])
        ax[2].set_title(f"{xcol} vs {ycol}  (n={len(sub):,})", color=C["text"])

    tc = tgt.dropna()
    ax[3].hist(tc, bins=40, color=C["purple"], alpha=0.85,
               edgecolor=C["bg"], linewidth=0.3)
    ax[3].axvline(tc.median(), color=C["red"], ls="--", lw=1.2,
                  label=f"Median={tc.median():.3f}")
    ax[3].axvline(tc.mean(), color=C["accent"], ls=":", lw=1.2,
                  label=f"Mean={tc.mean():.3f}")
    ax[3].set_title(f"Target: {meta['target_column']}  (n={len(tc):,})",
                    color=C["text"])
    ax[3].legend(fontsize=7)

    plotted = False
    for name, res in results.items():
        col = MODEL_CFG[name]["color"]
        yt, yp = np.array(res["y_test"]), np.array(res["y_pred"])
        yt, yp = _downsample_xy(yt, yp, max_points=2500, seed=SEED)
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
    ax[4].set_title("Predicted vs Actual — All Models", color=C["text"])
    ax[4].set_xlabel("Actual", color=C["dim"]); ax[4].set_ylabel("Predicted", color=C["dim"])
    if plotted: ax[4].legend(fontsize=6)

    sec = next((c for c in ("Baseline_FEV1_L", "BMI", "Weight", "Height")
                if c in df.columns and c != meta["target_column"]), None)
    if sec:
        d = pd.to_numeric(df[sec], errors="coerce").dropna()
        ax[5].hist(d, bins=40, color=C["teal"], alpha=0.85,
                   edgecolor=C["bg"], linewidth=0.3)
        ax[5].axvline(d.median(), color=C["red"], ls="--", lw=1.2,
                      label=f"Median={d.median():.3f}")
        ax[5].set_title(sec.replace("Baseline_","").replace("_"," ")
                        + f"  (n={len(d):,})", color=C["text"])
        ax[5].legend(fontsize=7)
    else:
        ax[5].set_visible(False)

    fig.suptitle("Dataset Overview", fontweight="bold", fontsize=13,
                 y=0.99, color=C["text"])
    fig.tight_layout(pad=1.5)
    return fig


def plot_feature_importance(results: dict, top_n=20):
    if not results:
        fig, ax = _make_fig(1,1,(6,4)); ax[0].axis("off"); return fig
    n = len(results)
    nrows, ncols = _grid_for(n)
    fig_w = 7 * ncols
    fig_h = max(5.2 * nrows, 6)
    fig, axes = _make_fig(nrows, ncols, (fig_w, fig_h))
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]; color = MODEL_CFG[name]["color"]
        fi = res["feature_importance"].head(top_n)
        labels = [l[:34]+"…" if len(l)>34 else l for l in fi.index]
        bars = ax.barh(range(len(fi)), fi.values, color=color, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(fi))); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel("Importance Score", color=C["dim"])
        ax.set_title(f"{name} — Top {len(fi)} [{res['fi_source']} importance]",
                     fontsize=9, color=color)
        if fi.values.max() > 0:
            for bar, val in zip(bars, fi.values):
                ax.text(val + fi.values.max()*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va="center", fontsize=6.5, color=C["text"])
    for ax in axes[n:]: ax.set_visible(False)
    fig.tight_layout(pad=2.5)
    return fig


def plot_predictions(results: dict, is_clf=True):
    if not results:
        fig, ax = _make_fig(1,1,(6,4)); ax[0].axis("off"); return fig
    n = len(results)
    if is_clf:
        nrows, ncols = _grid_for(n)
        fig, axes = _make_fig(nrows, ncols, (7*ncols, 5.4*nrows))
        for idx, (name, res) in enumerate(results.items()):
            ax = axes[idx]; color = MODEL_CFG[name]["color"]
            cm = res["cm"]; n_cls = cm.shape[0]
            im = ax.imshow(cm, cmap="Blues", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            thresh = cm.max() / 2
            for i in range(n_cls):
                for j in range(n_cls):
                    v = cm[i,j]
                    ax.text(j, i, f"{v:,}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if v > thresh else "#1a1a1a")
            ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
            ax.set_xticklabels([f"Pred {k}" for k in range(n_cls)], fontsize=8)
            ax.set_yticklabels([f"True {k}" for k in range(n_cls)], fontsize=8)
            roc = res.get("roc_auc")
            ax.set_title(f"{name} — Confusion Matrix\n"
                         f"Acc={res['accuracy']:.4f}"
                         + (f"  AUC={roc:.4f}" if roc else ""),
                         color=color, fontsize=9)
            ax.set_xlabel("Predicted Label", color=C["dim"])
            ax.set_ylabel("True Label", color=C["dim"])
        for ax in axes[n:]: ax.set_visible(False)
    else:
        fig, axes = _make_fig(n, 2, (12, 5*n))
        for ri, (name, res) in enumerate(results.items()):
            color = MODEL_CFG[name]["color"]
            yt, yp = np.array(res["y_test"]), np.array(res["y_pred"])
            ax_s = axes[ri*2]
            ax_s.scatter(yt, yp, alpha=0.3, s=10, color=color)
            mn, mx = yt.min(), yt.max()
            ax_s.plot([mn,mx],[mn,mx],"r--",lw=1.2,label="Ideal fit")
            ax_s.set_title(f"{name}  R²={res['r2']:.4f}", color=color)
            ax_s.set_xlabel("Actual",color=C["dim"]); ax_s.set_ylabel("Predicted",color=C["dim"])
            ax_s.legend(fontsize=7)
            ax_e = axes[ri*2+1]
            errs = yp - yt
            ax_e.hist(errs, bins=40, color=color, alpha=0.8, edgecolor=C["bg"])
            ax_e.axvline(0, color=C["red"], ls="--", lw=1.2)
            ax_e.set_title(f"{name}  MAE={res['mae']:.4f}", color=color)
            ax_e.set_xlabel("Error (Pred−Actual)",color=C["dim"])
            ax_e.set_ylabel("Count",color=C["dim"])
    fig.tight_layout(pad=2.0)
    return fig


def plot_residuals_cm(results: dict, is_clf=True):
    if not results:
        fig, ax = _make_fig(1,1,(6,4)); ax[0].axis("off"); return fig
    n = len(results)
    if is_clf:
        fig, axes = _make_fig(n, 1, (12.5, 4.2*n))
        for idx, (name, res) in enumerate(results.items()):
            color = MODEL_CFG[name]["color"]
            ax = axes[idx]; ax.axis("off")
            ax.text(0.02, 0.97, f"{name}  —  Classification Report",
                    transform=ax.transAxes, fontsize=12, color=color,
                    va="top", fontweight="bold")
            ax.text(0.02, 0.83, res["clf_report"],
                    transform=ax.transAxes, fontsize=9.5,
                    color=C["text"], va="top", fontfamily="monospace")
            summary = f"Accuracy: {res['accuracy']:.4f}"
            if res.get("roc_auc") is not None:
                summary += f"  |  ROC-AUC: {res['roc_auc']:.4f}"
            if res.get("avg_precision") is not None:
                summary += f"  |  Avg Precision: {res['avg_precision']:.4f}"
            if res.get("svm_subsampled"):
                summary += f"  |  SVM trained on {res['svm_train_size']:,} rows"
            ax.text(0.02, 0.05, summary, transform=ax.transAxes,
                    fontsize=9, color=C["green"], va="bottom", fontweight="bold")
    else:
        fig, axes = _make_fig(n, 2, (12, 5*n))
        for idx, (name, res) in enumerate(results.items()):
            color = MODEL_CFG[name]["color"]
            yt, yp = np.array(res["y_test"]), np.array(res["y_pred"])
            residuals = yp - yt
            ax_r = axes[idx*2]
            ax_r.scatter(yp, residuals, alpha=0.3, s=8, color=color)
            ax_r.axhline(0, color=C["red"], ls="--", lw=1.2)
            ax_r.set_xlabel("Predicted",color=C["dim"]); ax_r.set_ylabel("Residual",color=C["dim"])
            ax_r.set_title(f"{name} — Residuals vs Predicted", color=color)
            ax_q = axes[idx*2+1]
            (osm, osr), (slope, intercept, _) = sp_stats.probplot(residuals)
            ax_q.scatter(osm, osr, alpha=0.4, s=8, color=color)
            ax_q.plot(osm, slope*np.array(osm)+intercept,
                      color=C["red"], lw=1.5, label="Reference")
            ax_q.set_xlabel("Theoretical Quantiles",color=C["dim"])
            ax_q.set_ylabel("Sample Quantiles",color=C["dim"])
            ax_q.set_title(f"{name} — Q-Q Plot", color=color)
            ax_q.legend(fontsize=7)
    fig.tight_layout(pad=2.0)
    return fig


def plot_roc_pr(results: dict):
    fig, axes = _make_fig(1, 2, (12, 5.2))
    ax_r, ax_p = axes[0], axes[1]
    for name, res in results.items():
        if res.get("roc_data") is None: continue
        color = MODEL_CFG[name]["color"]
        fpr, tpr, _ = res["roc_data"]
        prec, rec, _ = res["pr_data"]
        ax_r.plot(fpr, tpr, color=color, lw=2,
                  label=f"{name}  AUC={res['roc_auc']:.4f}")
        ax_p.plot(rec, prec, color=color, lw=2,
                  label=f"{name}  AP={res['avg_precision']:.4f}")
    ax_r.plot([0,1],[0,1],color=C["dim"],ls="--",lw=1,label="Random")
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1.02)
    ax_r.set_xlabel("False Positive Rate",color=C["dim"])
    ax_r.set_ylabel("True Positive Rate",color=C["dim"])
    ax_r.set_title("ROC Curve — All Models",color=C["text"]); ax_r.legend(fontsize=8)
    ax_p.set_xlim(0,1); ax_p.set_ylim(0,1.02)
    ax_p.set_xlabel("Recall",color=C["dim"]); ax_p.set_ylabel("Precision",color=C["dim"])
    ax_p.set_title("Precision-Recall Curve — All Models",color=C["text"])
    ax_p.legend(fontsize=8)
    fig.tight_layout(pad=2.0)
    return fig


def parse_arduino_lidar_text(raw_text: str):
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    rows = []
    for i, ln in enumerate(lines, start=1):
        labeled = re.findall(r"S\s*([123])\s*:\s*([-+]?\d*\.?\d+)", ln, flags=re.IGNORECASE)
        if labeled:
            sensor_vals = {}
            for sid, val in labeled:
                sensor_vals[int(sid)] = float(val)
            if not all(k in sensor_vals for k in (1, 2, 3)):
                raise ValueError(f"Line {i}: expected S1/S2/S3 values, got '{ln}'")
            rows.append([sensor_vals[1], sensor_vals[2], sensor_vals[3]])
            continue

        nums = re.findall(r"[-+]?\d*\.?\d+", ln)
        if len(nums) < 3:
            raise ValueError(f"Line {i}: expected at least 3 numeric values, got '{ln}'")
        rows.append([float(nums[0]), float(nums[1]), float(nums[2])])
    if len(rows) < 2:
        raise ValueError("At least 2 lines are required (1 line = 1 second).")
    arr = np.asarray(rows, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Input contains non-finite values.")
    return arr


def compute_spirometry_from_lidar(dist_cm_arr, ball_mass_g=2.5, tube_length_cm=10.0,
                                  tube_diameter_cm=2.5, tube_volumes_cc=(600, 900, 1200), dt=1.0):
    dist_cm_arr = np.asarray(dist_cm_arr, dtype=float)
    if dist_cm_arr.ndim != 2 or dist_cm_arr.shape[1] != 3:
        raise ValueError("Expected sensor matrix with exactly 3 columns (S1, S2, S3).")
    if len(tube_volumes_cc) != 3:
        raise ValueError("tube_volumes_cc must contain 3 values for tube 1/2/3.")

    # Baseline distance per tube = farthest observed (ball at/near bottom).
    baseline = np.nanmax(dist_cm_arr, axis=0)
    minimum = np.nanmin(dist_cm_arr, axis=0)
    span = np.maximum(baseline - minimum, 1e-6)

    # Normalize each tube effort to [0,1] over the observed session.
    effort = np.clip((baseline - dist_cm_arr) / span, 0.0, 1.0)
    disp_cm = effort * float(tube_length_cm)

    disp_m = disp_cm / 100.0
    vel_mps = np.gradient(disp_m, dt, axis=0)
    acc_mps2 = np.gradient(vel_mps, dt, axis=0)

    radius_m = max(tube_diameter_cm / 200.0, 1e-4)
    area_m2 = np.pi * radius_m * radius_m
    mass_kg = max(ball_mass_g, 0.01) / 1000.0
    g = 9.80665

    # Pressure needed to support gravity + upward acceleration of ball.
    pressure_pa_tube = (mass_kg * np.maximum(g + acc_mps2, 0.0)) / area_m2
    pressure_cmh2o_tube = pressure_pa_tube / 98.0665

    caps_l = np.asarray(tube_volumes_cc, dtype=float) / 1000.0
    vol_tube_l = effort * caps_l
    vol_total_l = np.sum(vol_tube_l, axis=1)
    flow_lps = np.gradient(vol_total_l, dt)
    flow_lps = np.maximum(flow_lps, 0.0)

    # Robust aggregate pressure across tubes to reduce single-sensor noise.
    pressure_cmh2o = np.nanmedian(pressure_cmh2o_tube, axis=1)

    vol_l = np.maximum(vol_total_l, 0.0)
    idx_1s = min(len(vol_l) - 1, max(int(round(1.0 / max(dt, 1e-6))) - 1, 0))
    fev1 = float(vol_l[idx_1s])
    fvc = float(np.max(vol_l))
    ratio = float(fev1 / fvc) if fvc > 1e-8 else 0.0
    pef = float(np.max(flow_lps)) if len(flow_lps) else 0.0

    return {
        "pressure_cmh2o_ts": pressure_cmh2o,
        "flow_lps_ts": flow_lps,
        "volume_l_ts": vol_l,
        "fev1_l": max(fev1, 0.0),
        "fvc_l": max(fvc, 0.0),
        "fev1_fvc_ratio": float(np.clip(ratio, 0.0, 1.5)),
        "pef_lps": max(pef, 0.0),
        "n_seconds": int(len(flow_lps)),
        "tube_area_m2": float(area_m2),
    }


def build_patient_feature_row(meta, patient_inputs, spirometry_values):
    defaults = dict(meta.get("feature_defaults", {}))
    if not defaults:
        raise ValueError("Feature defaults are missing. Run pipeline first.")

    row_data = {c: np.nan for c in meta["feature_names"]}

    def _set_if_exists(col, val):
        if col in row_data and val is not None:
            row_data[col] = val

    _set_if_exists("Age", patient_inputs.get("age"))
    _set_if_exists("Gender", patient_inputs.get("gender"))
    _set_if_exists("Race", patient_inputs.get("race"))
    _set_if_exists("Height_cm", patient_inputs.get("height_cm"))
    _set_if_exists("Weight_kg", patient_inputs.get("weight_kg"))
    _set_if_exists("BMI", patient_inputs.get("bmi"))
    _set_if_exists("Smoking_Status", patient_inputs.get("smoking_status"))

    _set_if_exists("Baseline_FEV1_L", spirometry_values.get("fev1_l"))
    _set_if_exists("Baseline_FVC_L", spirometry_values.get("fvc_l"))
    _set_if_exists("Baseline_FEV1_FVC_Ratio", spirometry_values.get("fev1_fvc_ratio"))
    _set_if_exists("Baseline_PEF", spirometry_values.get("pef_lps"))
    _set_if_exists("Baseline_PEF_Ls", spirometry_values.get("pef_lps"))

    if "Lung_Index" in row_data:
        age = float(patient_inputs.get("age") or 0)
        fev1 = float(spirometry_values.get("fev1_l") or 0)
        row_data["Lung_Index"] = (fev1 / age) if age > 0 else 0.0

    Xp = pd.DataFrame([row_data], columns=meta["feature_names"])
    for c in Xp.columns:
        if pd.isna(Xp.at[0, c]):
            Xp.at[0, c] = defaults.get(c)

    cat_cols = meta.get("cat_cols", [])
    enc = meta.get("encoder")
    if cat_cols and enc is not None:
        Xp[cat_cols] = enc.transform(Xp[cat_cols])

    sc = meta.get("scaler")
    if sc is not None:
        Xp = pd.DataFrame(sc.transform(Xp), columns=meta["feature_names"])
    return Xp


def build_lung_health_text(spiro, predictions):
    fev1 = spiro["fev1_l"]
    fvc = spiro["fvc_l"]
    ratio = spiro["fev1_fvc_ratio"]
    pef = spiro["pef_lps"]
    p_med = float(np.median(spiro["pressure_cmh2o_ts"])) if len(spiro["pressure_cmh2o_ts"]) else 0.0

    if ratio < 0.70:
        pattern = "Obstructive pattern likely (FEV1/FVC below 0.70)."
        severity = "high" if ratio < 0.60 else "moderate"
    else:
        pattern = "No strong obstructive signal from FEV1/FVC ratio."
        severity = "low"

    pred_lines = []
    max_risk = 0.0
    top_model = None
    for name, out in predictions.items():
        risk = out.get("obstruction_prob")
        cls = out.get("pred_class")
        if risk is None:
            pred_lines.append(f"- {name}: class={cls}")
            continue
        pred_lines.append(f"- {name}: obstruction risk={risk*100:.1f}% (class={cls})")
        if risk > max_risk:
            max_risk = risk
            top_model = name

    tip_pack = [
        "Practice pursed-lip breathing for 5 minutes, twice daily.",
        "Avoid smoke, dust, and strong aerosol exposure.",
        "Do 20-30 minutes of moderate walking most days.",
        "Stay hydrated to reduce mucus thickness.",
        "Repeat spirometry weekly and track trend, not a single reading.",
    ]
    if severity in ("moderate", "high") or max_risk >= 0.50:
        tip_pack.insert(0, "Consult a pulmonologist for confirmatory spirometry and clinical exam.")
        tip_pack.append("If wheeze, chest tightness, or breathlessness worsens, seek urgent care.")

    lines = [
        "Patient Test Analysis",
        "=" * 58,
        f"FEV1: {fev1:.3f} L",
        f"FVC: {fvc:.3f} L",
        f"FEV1/FVC: {ratio:.3f}",
        f"PEF: {pef:.3f} L/s",
        f"Median Pressure: {p_med:.2f} cmH2O",
        "",
        f"Interpretation: {pattern}",
        "",
        "Model outputs:",
        *pred_lines,
        "",
        "Health assistant guidance:",
    ]
    lines.extend([f"- {t}" for t in tip_pack])
    if top_model is not None:
        lines.append("")
        lines.append(f"Highest estimated risk model: {top_model} ({max_risk*100:.1f}%).")
    return "\n".join(lines)


def plot_distributions(df_raw, numeric_cols, max_plots=9):
    priority = ["Baseline_FEV1_L","Baseline_FVC_L","Baseline_FEV1_FVC_Ratio",
                "Baseline_PEF_Ls","Baseline_FEF2575_Ls",
                "FEV1_Zscores_GLOBAL","FVC_Zscores_GLOBAL",
                "Age","Height","Weight","BMI"]
    ordered = ([c for c in priority if c in numeric_cols]
               + [c for c in numeric_cols if c not in priority])[:max_plots]
    n = len(ordered)
    if n == 0:
        fig, axes = _make_fig(1,1,(6,3)); axes[0].axis("off"); return fig
    pal = [C["accent"],C["teal"],C["orange"],C["purple"],
           C["green"],C["red"],C["pink"],"#93c5fd","#86efac"]
    nrows = max(1, int(np.ceil(n/3)))
    fig, flat = _make_fig(nrows, 3, (14, 3.8*nrows))
    for idx, col in enumerate(ordered):
        ax = flat[idx]
        data = pd.to_numeric(df_raw[col], errors="coerce").dropna()
        med = data.median()
        ax.hist(data, bins=40, color=pal[idx%len(pal)], alpha=0.85,
                edgecolor=C["bg"], linewidth=0.3)
        ax.axvline(med, color=C["red"], ls="--", lw=1.2,
                   label=f"Median={med:.3f}")
        ax.set_title(col.replace("_"," ").replace("Baseline ","")[:28],
                     fontsize=9, color=C["text"])
        ax.set_ylabel("Count", fontsize=8, color=C["dim"])
        ax.legend(fontsize=6.5, loc="upper right")
    for ax in flat[n:]: ax.set_visible(False)
    fig.suptitle("Key Column Distributions", fontweight="bold",
                 y=1.0, color=C["text"], fontsize=12)
    fig.tight_layout(pad=1.8)
    return fig


def plot_model_comparison(results: dict, is_clf: bool):
    if not results:
        fig, ax = _make_fig(1,1,(6,4)); ax[0].axis("off"); return fig
    fig, axes = _make_fig(1, 2, (14, 5.5))
    ax_b, ax_c = axes[0], axes[1]
    names = list(results.keys())
    colors = [MODEL_CFG[n]["color"] for n in names]
    N = len(names)
    if is_clf:
        metrics = [("accuracy","Accuracy"),("roc_auc","ROC-AUC"),
                   ("avg_precision","Avg Precision")]
    else:
        metrics = [("mae","MAE"),("rmse","RMSE"),("r2","R²")]
    n_m = len(metrics); x = np.arange(n_m); w = 0.7/N
    offsets = np.linspace(-(N-1)/2,(N-1)/2,N)*w
    all_vals = []
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name].get(k) or 0 for k,_ in metrics]
        all_vals.extend(vals)
        bars = ax_b.bar(x+offsets[i], vals, w*0.88,
                        label=name, color=color, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                      f"{val:.3f}", ha="center", va="bottom",
                      fontsize=6.5, color=C["text"])
    ax_b.set_xticks(x); ax_b.set_xticklabels([l for _,l in metrics], fontsize=9)
    ax_b.set_ylabel("Score", color=C["dim"])
    ax_b.set_title("Metrics Comparison — All Models", color=C["text"])
    ax_b.legend(fontsize=8)
    top = max(all_vals) if all_vals else 1.0
    ax_b.set_ylim(0, top*1.22)
    if is_clf:
        for name, color in zip(names, colors):
            if results[name].get("roc_data"):
                fpr, tpr, _ = results[name]["roc_data"]
                ax_c.plot(fpr, tpr, color=color, lw=2,
                          label=f"{name}  AUC={results[name]['roc_auc']:.4f}")
        ax_c.plot([0,1],[0,1],color=C["dim"],ls="--",lw=1)
        ax_c.set_title("ROC Curves Overlaid", color=C["text"])
        ax_c.set_xlabel("FPR", color=C["dim"]); ax_c.set_ylabel("TPR", color=C["dim"])
        ax_c.legend(fontsize=8)
    else:
        for name, color in zip(names, colors):
            errs = np.array(results[name]["y_pred"])-np.array(results[name]["y_test"])
            ax_c.hist(errs, bins=35, alpha=0.5, color=color, label=name)
        ax_c.axvline(0, color=C["red"], ls="--", lw=1.2)
        ax_c.set_title("Error Distribution", color=C["text"])
        ax_c.legend(fontsize=8)
    fig.tight_layout(pad=2.0)
    return fig


def build_conclusion(meta, results: dict):
    L = []
    def h(t):    L.append((t, "h"))
    def kv(k,v): L.append((f"  {k:<32}{v}", "kv"))
    def sep():   L.append(("─"*74, "sep"))
    def p(t):    L.append((t, "p"))
    def b():     L.append(("", "p"))
    def _tag(n): return n.lower().replace(" ","_").replace("-","_")

    L.append(("═"*74,"sep"))
    h("MODEL TRAINING SUMMARY — Spirometer Smart Health Monitor")
    p(f"  Target: {meta['target_column']}  •  NHANES 2007–2012  •  {len(results)} model(s)")
    L.append(("═"*74,"sep")); b()
    h("📂  DATASET"); sep()
    kv("File path:", meta["file_path"])
    kv("Total rows:", f"{meta['n_rows']:,}")
    kv("Total columns:", f"{meta['n_cols']:,}")
    kv("Feature count:", f"{meta['n_features']:,}")
    kv("Target column:", meta["target_column"])
    kv("Task type:", "Classification" if meta["is_clf"] else "Regression")
    kv("Train / Test:", f"{meta['train_size']:,} / {meta['test_size']:,}  (80/20)")
    ts = meta["target_series"].dropna()
    kv("Target mean:", f"{ts.mean():.4f}"); kv("Target std:", f"{ts.std():.4f}")
    if meta["is_clf"] and meta["class_dist"]:
        kv("Class distribution:", str(meta["class_dist"]))
    kv("Missing values (raw):", f"{meta['missing_raw']:,}")
    leaked = meta.get("leakage_dropped",0)
    if leaked: kv("Leakage cols dropped:", f"{leaked}  (ratio/diagnostic columns)")
    b()

    def _model_section(res, name):
        cfg = MODEL_CFG[name]; tag = _tag(name)
        sep(); L.append((f"{cfg['icon']}  {name.upper()}", tag)); b()
        pr = res["params"]; kv("Model class:", res["model_class"])
        for k in ("n_estimators","max_depth","learning_rate","subsample",
                  "colsample_bytree","class_weight","criterion","objective",
                  "eval_metric","kernel","C","gamma",
                  "hidden_layer_sizes","max_iter","early_stopping"):
            v = pr.get(k)
            if v is not None: kv(f"{k}:", str(v))
        if res.get("svm_subsampled"):
            kv("SVM training rows:", f"{res['svm_train_size']:,} (stratified subsample)")
        kv("Feature importance:", res["fi_source"])
        kv("Training time:", f"{res['train_seconds']:.2f} seconds")
        b(); L.append(("  📊  METRICS", tag))
        if meta["is_clf"]:
            acc = res["accuracy"]
            kv("Accuracy:", f"{acc:.4f}  ({acc*100:.2f}%)")
            roc = res.get("roc_auc")
            kv("ROC-AUC:", f"{roc:.4f}" if roc else "N/A")
            ap = res.get("avg_precision")
            kv("Avg Precision:", f"{ap:.4f}" if ap else "N/A")
            b(); p("  Classification Report:")
            L.append((res["clf_report"],"mono"))
        else:
            kv("MAE:",f"{res['mae']:.4f}"); kv("MSE:",f"{res['mse']:.4f}")
            kv("RMSE:",f"{res['rmse']:.4f}"); kv("R²:",f"{res['r2']:.4f}")
        b(); L.append(("  🏆  TOP 10 FEATURES", tag))
        for rank, (feat, score) in enumerate(res["feature_importance"].head(10).items(),1):
            short = feat[:42]+"…" if len(feat)>42 else feat
            kv(f"  #{rank:>2}  {short}", f"{score:.4f}")
        b()

    for name, res in results.items():
        _model_section(res, name)

    sep(); h("📈  KEY SPIROMETRY STATS (raw data)")
    for col in [c for c in ("Baseline_FEV1_L","Baseline_FVC_L","Baseline_FEV1_FVC_Ratio",
                             "Baseline_PEF_Ls","Baseline_FEF2575_Ls",
                             "FEV1_Zscores_GLOBAL","FVC_Zscores_GLOBAL",
                             "Age","Height","Weight","BMI")
                if c in meta["df_raw"].columns]:
        d = pd.to_numeric(meta["df_raw"][col], errors="coerce").dropna()
        kv(col, f"mean={d.mean():.3f}  std={d.std():.3f}  "
                f"min={d.min():.3f}  max={d.max():.3f}  n={len(d):,}")
    b()

    if len(results) > 1:
        sep(); h("⚖️  COMPARISON VERDICT"); b()
        pairs = ([("Best Accuracy:","accuracy",False),
                  ("Best ROC-AUC:","roc_auc",False),
                  ("Best Avg Precision:","avg_precision",False)]
                 if meta["is_clf"] else
                 [("Best R²:","r2",False),
                  ("Lowest MAE:","mae",True),
                  ("Lowest RMSE:","rmse",True)])
        for label, key, lower in pairs:
            scored = {n:(r.get(key) or 0) for n,r in results.items()}
            winner = (min(scored,key=scored.get) if lower
                      else max(scored,key=scored.get))
            vals_str = "  ".join(f"{n}={v:.4f}" for n,v in scored.items())
            L.append((f"  {label:<30}{winner}  [{vals_str}]", _tag(winner)))
        b(); h("⏱  TRAINING TIME")
        times = {n:r["train_seconds"] for n,r in results.items()}
        for n,t in times.items(): kv(f"{n}:",f"{t:.2f}s")
        kv("Fastest model:", min(times,key=times.get)); b()

    sep(); h("🩺  CLINICAL INTERPRETATION"); b()
    p("  FEV1 (Forced Expiratory Volume in 1 second) and FVC (Forced Vital")
    p("  Capacity) are primary spirometry markers of lung function.  An")
    p("  FEV1/FVC ratio below 0.70 is the standard GOLD criterion for")
    p("  obstructive lung disease (asthma, COPD)."); b()
    leaked = meta.get("leakage_dropped",0)
    if leaked:
        p(f"  ⚠  LEAKAGE PREVENTION:  {leaked} ratio/diagnostic columns excluded.")
        p("  Model predicts from genuine physiological features only."); b()
    for name, res in results.items():
        tag = _tag(name)
        if meta["is_clf"]:
            acc=res["accuracy"]; roc=res.get("roc_auc") or 0
            grade="Excellent" if acc>0.95 else "Good" if acc>0.80 else "Moderate"
            icon="✅" if grade=="Excellent" else "⚠️" if grade=="Good" else "❌"
            L.append((f"  {icon}  {name}: {grade} — Acc {acc*100:.2f}%  AUC {roc:.4f}", tag))
        else:
            r2=res.get("r2",0)
            grade="Excellent" if r2>0.85 else "Good" if r2>0.60 else "Moderate"
            icon="✅" if grade=="Excellent" else "⚠️"
            L.append((f"  {icon}  {name}: {grade} — R²={r2:.4f}", tag))
    b(); L.append(("═"*74,"sep"))
    return L


# ══════════════════════════════════════════════════════════════════════════════
#  MODERN UI  — CustomTkinter
# ══════════════════════════════════════════════════════════════════════════════

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Spirometer Monitor — 4-Model ML Dashboard  [v4]")
        self.configure(fg_color=C["bg"])
        try:
            self.state("zoomed")
        except Exception:
            self.geometry("1400x860")

        self._meta    = None
        self._results = {}

        self.v_path   = ctk.StringVar(
            value=r"D:\New folder (2)\NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv")
        self.v_target = ctk.StringVar(value="Obstruction")
        self.v_mode   = ctk.StringVar(value="Compare All")

        self.v_age = ctk.StringVar(value="30")
        self.v_gender = ctk.StringVar(value="Male")
        self.v_race = ctk.StringVar(value="Non-Hispanic White")
        self.v_height_cm = ctk.StringVar(value="170")
        self.v_weight_kg = ctk.StringVar(value="70")
        self.v_bmi = ctk.StringVar(value="24.2")
        self.v_smoke = ctk.StringVar(value="Never")
        self.v_ball_mass_g = ctk.StringVar(value="2.5")
        self.v_tube_len_cm = ctk.StringVar(value="10")
        self.v_tube_diam_cm = ctk.StringVar(value="2.5")
        self.v_tube1_cc = ctk.StringVar(value="600")
        self.v_tube2_cc = ctk.StringVar(value="900")
        self.v_tube3_cc = ctk.StringVar(value="1200")
        self.v_patient_model = ctk.StringVar(value="Compare All")

        self._pt_win = None
        self._pt_input = None
        self._pt_output = None
        self._pt_model_combo = None

        self._build()

    # ──────────────────────────────────────────────────────────────────────
    #  LAYOUT SKELETON
    # ──────────────────────────────────────────────────────────────────────

    def _build(self):
        self._topbar()
        body = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self._sidebar(body)
        self._main(body)

    # ──────────────────────────────────────────────────────────────────────
    #  TOPBAR
    # ──────────────────────────────────────────────────────────────────────

    def _topbar(self):
        bar = ctk.CTkFrame(self, fg_color=C["panel"], height=52, corner_radius=0)
        bar.pack(fill="x", side="top"); bar.pack_propagate(False)

        # left: icon + title
        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", padx=18, pady=8)
        ctk.CTkLabel(left, text="🫁", font=ctk.CTkFont(size=22)).pack(side="left")
        ctk.CTkLabel(left, text="  Spirometer Smart Health Monitor",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
                     text_color=C["text"]).pack(side="left")

        self._lbl_sub = ctk.CTkLabel(
            bar,
            text="XGBoost · Random Forest · SVM · MLP  •  NHANES 2007–2012",
            font=ctk.CTkFont(family=FONT_FAMILY, size=10),
            text_color=C["dim"])
        self._lbl_sub.pack(side="left", padx=20)

        # right: status pill + version badge
        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right", padx=20)
        self._dot_var = ctk.StringVar(value="● READY")
        self._lbl_dot = ctk.CTkLabel(
            right, textvariable=self._dot_var,
            font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"),
            text_color=C["green"])
        self._lbl_dot.pack(side="right")
        ctk.CTkLabel(right, text="v4  ",
                     font=ctk.CTkFont(size=9),
                     text_color=C["dim"]).pack(side="right")

        # bottom divider
        ctk.CTkFrame(self, fg_color=C["border"], height=1,
                     corner_radius=0).pack(fill="x")

    # ──────────────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ──────────────────────────────────────────────────────────────────────

    def _sidebar(self, parent):
        sb = ctk.CTkScrollableFrame(
            parent, width=230, fg_color=C["panel"],
            scrollbar_button_color=C["border"],
            scrollbar_button_hover_color=C["card2"],
            corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew", padx=(0,0), pady=0)

        def _sec(icon, txt):
            f = ctk.CTkFrame(sb, fg_color="transparent")
            f.pack(fill="x", padx=12, pady=(16, 4))
            ctk.CTkLabel(f, text=icon+"  "+txt,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"),
                         text_color=C["accent"]).pack(side="left")

        def _label(txt, dim=True):
            ctk.CTkLabel(sb, text=txt,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=9),
                         text_color=C["dim"] if dim else C["text"],
                         anchor="w").pack(fill="x", padx=14, pady=(2,0))

        def _divider():
            ctk.CTkFrame(sb, fg_color=C["border"], height=1,
                         corner_radius=0).pack(fill="x", pady=6)

        # ── Dataset ──────────────────────────────────────────────────────
        _sec("📂", "DATASET")
        _label("CSV file path:")

        path_card = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=8)
        path_card.pack(fill="x", padx=12, pady=(2,6))
        self._lbl_path = ctk.CTkLabel(
            path_card, textvariable=self.v_path,
            font=ctk.CTkFont(family=FONT_FAMILY, size=8),
            text_color=C["accent"], wraplength=190,
            justify="left", cursor="hand2", anchor="w")
        self._lbl_path.pack(fill="x", padx=8, pady=6)
        self._lbl_path.bind("<Button-1>", self._browse)
        self._lbl_path.bind("<Enter>",
            lambda e: self._lbl_path.configure(text_color=C["text"]))
        self._lbl_path.bind("<Leave>",
            lambda e: self._lbl_path.configure(text_color=C["accent"]))

        # ── Target ───────────────────────────────────────────────────────
        _sec("◎", "TARGET COLUMN")
        self._cb_tgt = ctk.CTkComboBox(
            sb, variable=self.v_target, width=206,
            fg_color=C["card"], border_color=C["border"],
            button_color=C["accent"], button_hover_color=C["accent2"],
            dropdown_fg_color=C["card"], dropdown_text_color=C["text"],
            dropdown_hover_color=C["card2"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=10),
            text_color=C["text"])
        self._cb_tgt.pack(padx=12, pady=(2,8))

        # ── Model ─────────────────────────────────────────────────────────
        _sec("◈", "MODEL")
        mode_options = list(MODEL_CFG.keys()) + ["Compare All"]
        self._rb_frames = {}
        for lbl in mode_options:
            col = MODEL_CFG[lbl]["ctk_color"] if lbl in MODEL_CFG else C["teal"]
            rb = ctk.CTkRadioButton(
                sb, text=lbl, variable=self.v_mode, value=lbl,
                font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"),
                text_color=col,
                fg_color=col, hover_color=col,
                border_color=C["border"])
            rb.pack(anchor="w", padx=18, pady=3)
            self._rb_frames[lbl] = rb

        # ── Trained Params ────────────────────────────────────────────────
        _sec("⚙", "TRAINED PARAMS")
        self._pf = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=8)
        self._pf.pack(fill="x", padx=12, pady=(2,8))
        self._show_params({})

        _divider()

        # ── Run Button ────────────────────────────────────────────────────
        self._btn = ctk.CTkButton(
            sb, text="▶   Run Pipeline",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            fg_color=C["green"], hover_color="#059669",
            text_color="#052e16", height=44, corner_radius=10,
            command=self._on_run)
        self._btn.pack(fill="x", padx=12, pady=(4,4))

        ctk.CTkButton(
            sb, text="💾  Save All Plots",
            font=ctk.CTkFont(family=FONT_FAMILY, size=10),
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["text"], height=32, corner_radius=8,
            command=self._save).pack(fill="x", padx=12, pady=(0,8))

        ctk.CTkButton(
            sb, text="🫁  Open Patient Test Lab",
            font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"),
            fg_color=C["accent"], hover_color=C["accent2"],
            text_color=C["text"], height=34, corner_radius=8,
            command=self._open_patient_test_window).pack(fill="x", padx=12, pady=(0,8))

        _divider()

        # ── Dataset Info ──────────────────────────────────────────────────
        _sec("ℹ", "DATASET INFO")
        info_card = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=8)
        info_card.pack(fill="x", padx=12, pady=(2,12))
        self._info = {}
        for k in ("Rows","Columns","Features","Task",
                  "Train Size","Test Size","Missing (raw)","Leakage Cols"):
            row = ctk.CTkFrame(info_card, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=2)
            ctk.CTkLabel(row, text=k+":",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=9),
                         text_color=C["dim"], width=100, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(row, text="—",
                               font=ctk.CTkFont(family=FONT_FAMILY, size=9, weight="bold"),
                               text_color=C["text"], anchor="w")
            lbl.pack(side="left")
            self._info[k] = lbl

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN CONTENT AREA
    # ──────────────────────────────────────────────────────────────────────

    def _main(self, parent):
        main = ctk.CTkFrame(parent, fg_color=C["bg"], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # ── Metric Cards strip ────────────────────────────────────────────
        self._cards_outer = ctk.CTkFrame(main, fg_color=C["bg"], corner_radius=0)
        self._cards_outer.grid(row=0, column=0, sticky="ew", padx=10, pady=(8,0))
        self._crows = {}
        for mname, cfg in MODEL_CFG.items():
            row_frame = ctk.CTkFrame(self._cards_outer, fg_color="transparent")
            # model name label
            ctk.CTkLabel(row_frame, text=mname,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"),
                         text_color=cfg["color"], width=130, anchor="e").pack(
                             side="left", padx=(0, 8))
            cards = {}
            # Only show relevant metrics — dynamically determined at update time
            # We always build all 6 cards but hide regression ones for clf
            for met, tip in [
                ("Accuracy",     "Fraction of correct predictions.  1.0 = perfect."),
                ("ROC-AUC",      "Area Under ROC Curve.  1.0 = perfect, 0.5 = random."),
                ("Avg Precision","Area under Precision-Recall curve.  Higher = better."),
                ("MAE",          "Mean Absolute Error.  Lower = better. (regression only)"),
                ("RMSE",         "Root Mean Squared Error.  Lower = better. (regression only)"),
                ("R²",           "Variance explained by model.  1.0 = perfect. (regression only)"),
            ]:
                cf = ctk.CTkFrame(row_frame, fg_color=C["card"],
                                  corner_radius=10, width=90, height=60)
                cf.pack(side="left", padx=4, pady=3)
                cf.pack_propagate(False)
                ctk.CTkLabel(cf, text=met,
                             font=ctk.CTkFont(family=FONT_FAMILY, size=8),
                             text_color=C["dim"]).pack(pady=(6,0))
                vl = ctk.CTkLabel(cf, text="—",
                                  font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
                                  text_color=C["dim"])
                vl.pack(pady=(0,6))
                cards[met] = (cf, vl)
                self._add_tooltip(cf, tip)
            self._crows[mname] = {"frame": row_frame, "cards": cards}

        # ── Tab Bar ───────────────────────────────────────────────────────
        self._nb = ctk.CTkTabview(
            main, fg_color=C["card"], segmented_button_fg_color=C["panel"],
            segmented_button_selected_color=C["accent"],
            segmented_button_selected_hover_color=C["accent2"],
            segmented_button_unselected_color=C["panel"],
            segmented_button_unselected_hover_color=C["card2"],
            text_color=C["dim"], text_color_disabled=C["dim"],
            corner_radius=12)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)

        tab_names = [
            "📈 Overview", "🏆 Feature Importance", "🎯 Predictions",
            "📉 Residuals/CM", "📡 ROC/PR Curves", "📊 Distributions",
            "⚖️ Model Comparison", "📋 Conclusion",
        ]
        self._tabs = {}
        for t in tab_names:
            self._nb.add(t)
            self._tabs[t] = self._nb.tab(t)

        # Conclusion needs a text widget
        cl = self._tabs["📋 Conclusion"]
        cl.grid_columnconfigure(0, weight=1); cl.grid_rowconfigure(0, weight=1)

        txt_frame = ctk.CTkFrame(cl, fg_color=C["card"], corner_radius=8)
        txt_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        txt_frame.grid_columnconfigure(0, weight=1)
        txt_frame.grid_rowconfigure(0, weight=1)

        self._txt = tk.Text(
            txt_frame, bg=C["card"], fg=C["text"],
            font=("Consolas", 9), wrap="word", relief="flat",
            padx=16, pady=12, state="disabled", cursor="arrow",
            insertbackground=C["text"], selectbackground=C["border"])
        vsb = ctk.CTkScrollbar(txt_frame, command=self._txt.yview,
                               button_color=C["border"],
                               button_hover_color=C["card2"])
        self._txt.configure(yscrollcommand=vsb.set)
        self._txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Text tags
        for tag, cfg_t in [
            ("h",    dict(font=("Consolas",11,"bold"), foreground=C["accent"])),
            ("kv",   dict(foreground=C["text"])),
            ("sep",  dict(foreground=C["border"])),
            ("p",    dict(foreground=C["dim"])),
            ("mono", dict(font=("Consolas",8), foreground=C["text"])),
        ]:
            self._txt.tag_configure(tag, **cfg_t)
        for name, cfg in MODEL_CFG.items():
            tag = name.lower().replace(" ","_").replace("-","_")
            self._txt.tag_configure(
                tag, font=("Consolas",9,"bold"), foreground=cfg["color"])

    def _open_patient_test_window(self):
        if self._pt_win is not None and self._pt_win.winfo_exists():
            self._pt_win.focus_force()
            return

        w = ctk.CTkToplevel(self)
        w.title("Patient Test Lab — Arduino LiDAR")
        w.geometry("1180x760")
        w.configure(fg_color=C["bg"])
        w.grid_columnconfigure(0, weight=1)
        w.grid_columnconfigure(1, weight=1)
        w.grid_rowconfigure(0, weight=1)
        self._pt_win = w

        left = ctk.CTkScrollableFrame(
            w, fg_color=C["card"], corner_radius=10,
            scrollbar_button_color=C["border"],
            scrollbar_button_hover_color=C["card2"])
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.grid_columnconfigure(1, weight=1)

        right = ctk.CTkFrame(w, fg_color=C["card"], corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="Human Test Input (separate from main dashboard)",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
                     text_color=C["text"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 8))

        def row(lbl, var, r, values=None):
            ctk.CTkLabel(left, text=lbl, text_color=C["dim"],
                         font=ctk.CTkFont(size=10)).grid(row=r, column=0, sticky="w", padx=12, pady=4)
            if values is None:
                e = ctk.CTkEntry(left, textvariable=var, fg_color=C["panel"], border_color=C["border"], text_color=C["text"])
            else:
                e = ctk.CTkComboBox(left, variable=var, values=values,
                                    fg_color=C["panel"], border_color=C["border"],
                                    button_color=C["accent"], button_hover_color=C["accent2"],
                                    dropdown_fg_color=C["panel"], dropdown_text_color=C["text"])
            e.grid(row=r, column=1, sticky="ew", padx=(0, 12), pady=4)
            return e

        row("Age", self.v_age, 1)
        row("Gender", self.v_gender, 2, ["Male", "Female", "Other"])
        row("Race", self.v_race, 3, [
            "Mexican American", "Other Hispanic", "Non-Hispanic White",
            "Non-Hispanic Black", "Other/Multi"
        ])
        row("Height (cm)", self.v_height_cm, 4)
        row("Weight (kg)", self.v_weight_kg, 5)
        row("BMI", self.v_bmi, 6)
        row("Smoking Status", self.v_smoke, 7, ["Never", "Former", "Current", "Unknown"])
        row("Ball Mass (g)", self.v_ball_mass_g, 8)
        row("Tube Length (cm)", self.v_tube_len_cm, 9)
        row("Tube Diameter (cm)", self.v_tube_diam_cm, 10)
        row("Tube-1 Full Volume (cc)", self.v_tube1_cc, 11)
        row("Tube-2 Full Volume (cc)", self.v_tube2_cc, 12)
        row("Tube-3 Full Volume (cc)", self.v_tube3_cc, 13)
        self._pt_model_combo = row("Model (Patient Test)", self.v_patient_model, 14,
                                   list(MODEL_CFG.keys()) + ["Compare All"])
        if self._results:
            mvals = list(self._results.keys())
            if len(mvals) > 1:
                mvals.append("Compare All")
            self._pt_model_combo.configure(values=mvals)
            if self.v_patient_model.get() not in mvals:
                self.v_patient_model.set("Compare All" if "Compare All" in mvals else mvals[0])

        ctk.CTkButton(left, text="Analyze Test", height=38,
                      fg_color=C["green"], hover_color="#059669", text_color="#052e16",
                      command=self._run_patient_test_window).grid(
            row=15, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 8)
        )

        ctk.CTkLabel(left, text="Paste Arduino lines. Supported format: S1:56  S2:87  S3:125",
                     text_color=C["dim"], font=ctk.CTkFont(size=10)).grid(
            row=16, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 4)
        )

        self._pt_input = ctk.CTkTextbox(left, fg_color=C["panel"], border_color=C["border"],
                                        text_color=C["text"], height=190)
        self._pt_input.grid(row=17, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))
        self._pt_input.insert("1.0", "S1:56  S2:87  S3:125\nS1:67  S2:92  S3:124\nS1:82  S2:89  S3:128")
        self._pt_input.bind("<Control-Return>", lambda _e: self._run_patient_test_window())

        ctk.CTkLabel(right, text="Lung Health Conclusion", text_color=C["text"],
                     font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 6)
        )
        self._pt_output = ctk.CTkTextbox(right, fg_color=C["panel"], border_color=C["border"],
                                         text_color=C["text"])
        self._pt_output.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _run_patient_test_common(self, raw_text: str, model_choice: str):
        if not self._results or not self._meta:
            raise ValueError("Please run the training pipeline first.")

        arr = parse_arduino_lidar_text(raw_text)

        age = float(self.v_age.get())
        height_cm = float(self.v_height_cm.get())
        weight_kg = float(self.v_weight_kg.get())
        bmi = float(self.v_bmi.get()) if self.v_bmi.get().strip() else (weight_kg / ((height_cm / 100.0) ** 2))

        spiro = compute_spirometry_from_lidar(
            arr,
            ball_mass_g=float(self.v_ball_mass_g.get()),
            tube_length_cm=float(self.v_tube_len_cm.get()),
            tube_diameter_cm=float(self.v_tube_diam_cm.get()),
            tube_volumes_cc=(float(self.v_tube1_cc.get()), float(self.v_tube2_cc.get()), float(self.v_tube3_cc.get())),
            dt=1.0,
        )

        patient_inputs = {
            "age": age,
            "gender": self.v_gender.get(),
            "race": self.v_race.get(),
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "smoking_status": self.v_smoke.get(),
        }

        Xp = build_patient_feature_row(self._meta, patient_inputs, spiro)
        preds = {}

        names = list(self._results.keys())
        if model_choice and model_choice != "Compare All":
            if model_choice not in self._results:
                raise ValueError(f"Selected model '{model_choice}' is not trained in current run.")
            names = [model_choice]

        for name in names:
            res = self._results[name]
            mdl = res["model"]
            yhat = mdl.predict(Xp)[0]
            out = {"pred_class": int(yhat) if str(yhat).isdigit() else yhat}
            if hasattr(mdl, "predict_proba"):
                pr = mdl.predict_proba(Xp)
                if pr.shape[1] >= 2:
                    out["obstruction_prob"] = float(pr[0, 1])
            preds[name] = out

        return build_lung_health_text(spiro, preds)

    def _run_patient_test_window(self):
        try:
            raw = self._pt_input.get("1.0", "end").strip() if self._pt_input else ""
            report = self._run_patient_test_common(raw, self.v_patient_model.get())
            if self._pt_output is not None:
                self._pt_output.delete("1.0", "end")
                self._pt_output.insert("1.0", report)
        except Exception as ex:
            messagebox.showerror("Patient Test Error", str(ex))

    def _build_patient_test_tab(self):
        tab = self._tabs["🫁 Patient Test"]
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(tab, fg_color=C["card"], corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        left.grid_columnconfigure(1, weight=1)

        right = ctk.CTkFrame(tab, fg_color=C["card"], corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="Human Test Input (Arduino LiDAR)",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
                     text_color=C["text"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 8))

        def row(lbl, var, r, values=None):
            ctk.CTkLabel(left, text=lbl, text_color=C["dim"],
                         font=ctk.CTkFont(size=10)).grid(row=r, column=0, sticky="w", padx=12, pady=4)
            if values is None:
                e = ctk.CTkEntry(left, textvariable=var, fg_color=C["panel"], border_color=C["border"], text_color=C["text"])
            else:
                e = ctk.CTkComboBox(left, variable=var, values=values,
                                    fg_color=C["panel"], border_color=C["border"],
                                    button_color=C["accent"], button_hover_color=C["accent2"],
                                    dropdown_fg_color=C["panel"], dropdown_text_color=C["text"])
            e.grid(row=r, column=1, sticky="ew", padx=(0, 12), pady=4)

        row("Age", self.v_age, 1)
        row("Gender", self.v_gender, 2, ["Male", "Female", "Other"])
        row("Race", self.v_race, 3, [
            "Mexican American", "Other Hispanic", "Non-Hispanic White",
            "Non-Hispanic Black", "Other/Multi"
        ])
        row("Height (cm)", self.v_height_cm, 4)
        row("Weight (kg)", self.v_weight_kg, 5)
        row("BMI", self.v_bmi, 6)
        row("Smoking Status", self.v_smoke, 7, ["Never", "Former", "Current", "Unknown"])
        row("Ball Mass (g)", self.v_ball_mass_g, 8)
        row("Tube Length (cm)", self.v_tube_len_cm, 9)
        row("Tube Diameter (cm)", self.v_tube_diam_cm, 10)

        ctk.CTkLabel(left, text="Paste Arduino lines (3 values per line; 1 line = 1 second)",
                     text_color=C["dim"], font=ctk.CTkFont(size=10)).grid(
            row=11, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 4)
        )

        self._arduino_txt = ctk.CTkTextbox(left, fg_color=C["panel"], border_color=C["border"],
                                           text_color=C["text"], height=220)
        self._arduino_txt.grid(row=12, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 8))
        self._arduino_txt.insert("1.0", "8.1 8.0 8.2\n7.9 7.8 7.9\n7.3 7.2 7.4\n6.8 6.7 6.9")

        ctk.CTkButton(left, text="Analyze Patient Test", height=36,
                      fg_color=C["green"], hover_color="#059669", text_color="#052e16",
                      command=self._run_patient_test).grid(
            row=13, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12)
        )

        ctk.CTkLabel(right, text="Lung Health Assessment", text_color=C["text"],
                     font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 6)
        )
        self._patient_out = ctk.CTkTextbox(right, fg_color=C["panel"], border_color=C["border"],
                                           text_color=C["text"])
        self._patient_out.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _run_patient_test(self):
        if not self._results or not self._meta:
            messagebox.showinfo("Run model first", "Please run the training pipeline first.")
            return
        try:
            raw = self._arduino_txt.get("1.0", "end").strip()
            arr = parse_arduino_lidar_text(raw)

            age = float(self.v_age.get())
            height_cm = float(self.v_height_cm.get())
            weight_kg = float(self.v_weight_kg.get())
            bmi = float(self.v_bmi.get()) if self.v_bmi.get().strip() else (weight_kg / ((height_cm / 100.0) ** 2))

            spiro = compute_spirometry_from_lidar(
                arr,
                ball_mass_g=float(self.v_ball_mass_g.get()),
                tube_length_cm=float(self.v_tube_len_cm.get()),
                tube_diameter_cm=float(self.v_tube_diam_cm.get()),
                dt=1.0,
            )

            patient_inputs = {
                "age": age,
                "gender": self.v_gender.get(),
                "race": self.v_race.get(),
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "bmi": bmi,
                "smoking_status": self.v_smoke.get(),
            }

            Xp = build_patient_feature_row(self._meta, patient_inputs, spiro)
            preds = {}
            for name, res in self._results.items():
                mdl = res["model"]
                yhat = mdl.predict(Xp)[0]
                out = {"pred_class": int(yhat) if str(yhat).isdigit() else yhat}
                if hasattr(mdl, "predict_proba"):
                    pr = mdl.predict_proba(Xp)
                    if pr.shape[1] >= 2:
                        out["obstruction_prob"] = float(pr[0, 1])
                preds[name] = out

            report = build_lung_health_text(spiro, preds)
            self._patient_out.delete("1.0", "end")
            self._patient_out.insert("1.0", report)
            self._nb.set("🫁 Patient Test")
        except Exception as ex:
            messagebox.showerror("Patient Test Error", str(ex))

    # ──────────────────────────────────────────────────────────────────────
    #  TOOLTIP  (tk.Toplevel — works with ctk windows)
    # ──────────────────────────────────────────────────────────────────────

    def _add_tooltip(self, widget, text):
        if not text: return
        tip = None
        def _enter(e):
            nonlocal tip
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{e.x_root+14}+{e.y_root+14}")
            tk.Label(tip, text=text,
                     font=(FONT_FAMILY, 8), bg=C["card2"], fg=C["text"],
                     relief="flat", borderwidth=0, padx=10, pady=6,
                     justify="left", wraplength=260).pack()
        def _leave(e):
            nonlocal tip
            if tip: tip.destroy(); tip=None
        widget.bind("<Enter>", _enter, add="+")
        widget.bind("<Leave>", _leave, add="+")

    # ──────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _browse(self, _=None):
        p = filedialog.askopenfilename(
            title="Select CSV", filetypes=[("CSV","*.csv"),("All","*.*")])
        if p:
            self.v_path.set(p)
            try:
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                self._cb_tgt.configure(values=cols)
            except Exception: pass

    def _dot(self, txt, color=None):
        def _do():
            self._dot_var.set(txt)
            self._lbl_dot.configure(text_color=color or C["green"])
        self.after(0, _do)

    def _show_params(self, params):
        for w in self._pf.winfo_children():
            w.destroy()
        skip = {
            "num_class","verbosity","n_jobs","base_score","booster","callbacks",
            "early_stopping_rounds","enable_categorical","feature_types",
            "grow_policy","importance_type","interaction_constraints","max_bin",
            "max_cat_threshold","max_cat_to_onehot","max_delta_step","max_leaves",
            "min_child_weight","missing","monotone_constraints","multi_strategy",
            "nthread","num_parallel_tree","reg_alpha","reg_lambda",
            "sampling_method","scale_pos_weight","tree_method",
            "validate_parameters","device","min_samples_split","min_samples_leaf",
            "min_weight_fraction_leaf","max_features","max_leaf_nodes",
            "min_impurity_decrease","bootstrap","oob_score","warm_start",
            "ccp_alpha","max_samples","verbose","activation","solver","alpha",
            "batch_size","learning_rate","power_t","shuffle","tol",
            "n_iter_no_change","beta_1","beta_2","epsilon","momentum",
            "nesterovs_momentum","validation_fraction","t0","cache_size",
            "decision_function_shape","break_ties","coef0","degree","shrinking",
        }
        shown = {k:v for k,v in params.items()
                 if v is not None and k not in skip}
        if not shown:
            ctk.CTkLabel(self._pf, text="Train a model to see params",
                         font=ctk.CTkFont(size=8),
                         text_color=C["dim"]).pack(padx=8, pady=6)
            return
        for k, v in list(shown.items())[:12]:
            row = ctk.CTkFrame(self._pf, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=1)
            ctk.CTkLabel(row, text=f"{k}:",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=8),
                         text_color=C["dim"], width=110, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=str(v)[:18],
                         font=ctk.CTkFont(family=FONT_FAMILY, size=8, weight="bold"),
                         text_color=C["green"], anchor="w").pack(side="left")

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
            color = (C["orange"] if k == "Task"
                     else C["red"] if k == "Leakage Cols" and leaked > 0
                     else C["text"])
            self._info[k].configure(text=v, text_color=color)

    def _show_card_rows(self, results: dict, is_clf: bool):
        """Show rows for trained models; show only relevant metric cards."""
        for name, data in self._crows.items():
            data["frame"].pack_forget()

        # metrics to show depend on task type
        clf_metrics = {"Accuracy", "ROC-AUC", "Avg Precision"}
        reg_metrics = {"MAE", "RMSE", "R²"}
        hide_metrics = reg_metrics if is_clf else clf_metrics

        for name in results:
            if name not in self._crows: continue
            data = self._crows[name]
            # toggle card visibility
            for met, (cf, vl) in data["cards"].items():
                if met in hide_metrics:
                    cf.pack_forget()
                else:
                    # re-pack if not already packed
                    cf.pack(side="left", padx=4, pady=3)
            data["frame"].pack(fill="x", pady=3)

    def _upd_cards(self, res, model_name: str, is_clf: bool):
        cards = self._crows[model_name]["cards"]
        color = MODEL_CFG[model_name]["color"]
        if is_clf:
            mp = {"Accuracy":     ("accuracy",     color),
                  "ROC-AUC":      ("roc_auc",      C["accent"]),
                  "Avg Precision":("avg_precision", C["purple"])}
        else:
            mp = {"MAE":  ("mae",  C["orange"]),
                  "RMSE": ("rmse", C["red"]),
                  "R²":   ("r2",   C["green"])}
        for met, (key, col) in mp.items():
            _, vl = cards[met]
            val = res.get(key)
            if val is not None:
                vl.configure(text=f"{val:.4f}", text_color=col)
            else:
                vl.configure(text="—", text_color=C["dim"])

    def _reset_cards(self):
        for data in self._crows.values():
            for _, (_, vl) in data["cards"].items():
                vl.configure(text="—", text_color=C["dim"])

    # ──────────────────────────────────────────────────────────────────────
    #  RUN PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def _on_run(self):
        self._btn.configure(state="disabled", text="⏳  Running…")
        self._reset_cards()
        self._dot("● LOADING", C["orange"])
        threading.Thread(target=self._pipeline, daemon=True).start()

    def _pipeline(self):
        try:
            path   = self.v_path.get().strip()
            target = self.v_target.get().strip()
            mode   = self.v_mode.get()

            self._dot("● PREPROCESSING", C["orange"])
            X_tr, X_te, y_tr, y_te, meta = load_and_preprocess(path, target)
            self._meta = meta
            n_cls = int(y_tr.nunique()) if meta["is_clf"] else 0

            results = {}
            for name in MODEL_CFG:
                if mode not in (name, "Compare All"): continue
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
            self.after(0, lambda: self._dot("● ERROR", C["red"]))
            self.after(0, lambda: self._btn.configure(
                state="normal", text="▶   Run Pipeline"))

    def _post_run(self):
        meta    = self._meta
        results = self._results
        is_clf  = meta["is_clf"]

        self._upd_info(meta)
        if results:
            self._show_params(next(reversed(results.values()))["params"])

        self._show_card_rows(results, is_clf)
        for name, res in results.items():
            self._upd_cards(res, name, is_clf)

        self._dot("● RENDERING", C["accent"])
        self._render(meta, results, is_clf)

        mode = self.v_mode.get()
        self._lbl_sub.configure(
            text=f"{mode}  •  Target: {meta['target_column']}  •  NHANES 2007–2012")
        self._dot("● READY", C["green"])
        self._btn.configure(state="normal", text="▶   Run Pipeline")

    def _embed_in_tab(self, fig, tab_name: str):
        tab = self._tabs[tab_name]
        _embed(fig, tab)

    def _render(self, meta, results, is_clf):
        self._embed_in_tab(plot_overview(meta, results),           "📈 Overview")
        self._embed_in_tab(plot_feature_importance(results),       "🏆 Feature Importance")
        self._embed_in_tab(plot_predictions(results, is_clf),      "🎯 Predictions")
        self._embed_in_tab(plot_residuals_cm(results, is_clf),     "📉 Residuals/CM")

        if is_clf:
            self._embed_in_tab(plot_roc_pr(results), "📡 ROC/PR Curves")
        else:
            tab = self._tabs["📡 ROC/PR Curves"]
            for w in tab.winfo_children(): w.destroy()
            ctk.CTkLabel(tab,
                text="ROC / PR curves are only available\nfor classification tasks.",
                font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                text_color=C["dim"]).pack(expand=True)

        self._embed_in_tab(
            plot_distributions(meta["df_raw"], meta["numeric_raw_cols"]),
            "📊 Distributions")

        if len(results) > 1:
            self._embed_in_tab(plot_model_comparison(results, is_clf),
                               "⚖️ Model Comparison")
        else:
            tab = self._tabs["⚖️ Model Comparison"]
            for w in tab.winfo_children(): w.destroy()
            ctk.CTkLabel(tab,
                text="Select 'Compare All' to see side-by-side model comparison.",
                font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                text_color=C["dim"]).pack(expand=True)

        lines = build_conclusion(meta, results)
        t = self._txt
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, tag in lines:
            t.insert("end", text+"\n", tag)
        t.configure(state="disabled")
        t.yview_moveto(0)

    # ──────────────────────────────────────────────────────────────────────
    #  SAVE
    # ──────────────────────────────────────────────────────────────────────

    def _save(self):
        if self._meta is None:
            messagebox.showinfo("No results", "Train a model first.")
            return
        folder = filedialog.askdirectory(title="Select folder to save plots")
        if not folder: return
        meta    = self._meta
        results = self._results
        is_clf  = meta["is_clf"]
        figs = {
            "overview.png":           plot_overview(meta, results),
            "feature_importance.png": plot_feature_importance(results),
            "predictions.png":        plot_predictions(results, is_clf),
            "residuals_cm.png":       plot_residuals_cm(results, is_clf),
            "distributions.png":      plot_distributions(meta["df_raw"],
                                          meta["numeric_raw_cols"]),
        }
        if is_clf:
            figs["roc_pr.png"] = plot_roc_pr(results)
        if len(results) > 1:
            figs["model_comparison.png"] = plot_model_comparison(results, is_clf)
        for fname, fig in figs.items():
            fig.savefig(os.path.join(folder, fname),
                        bbox_inches="tight", dpi=150, facecolor=C["plot_bg"])
        messagebox.showinfo("Saved", f"All plots saved to:\n{folder}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    try:
        cols = pd.read_csv(app.v_path.get(), nrows=0).columns.tolist()
        app._cb_tgt.configure(values=cols)
    except Exception:
        pass
    app.mainloop()
