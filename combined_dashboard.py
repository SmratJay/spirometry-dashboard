"""
=============================================================================
  Spirometer Smart Health Monitor
  XGBoost + Random Forest — Unified ML Dashboard
  Final Year Research Project — NHANES 2007-2012
  
  Every metric, label, stat, and plot value is computed live from
  the loaded dataset and trained model objects. Nothing is hardcoded.
=============================================================================
"""

import os, threading, traceback, warnings

import numpy as np
import pandas as pd
import scipy.stats as sp_stats

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score,
)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── silence non-critical warnings ────────────────────────────────────────────
for cat in (UserWarning, FutureWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=cat)

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  —  single source of truth for every colour / font
# ══════════════════════════════════════════════════════════════════════════════
T = dict(
    bg="#0d1117", card="#161b22", sidebar="#010409",
    border="#30363d", input_bg="#21262d",
    blue="#58a6ff", green="#3fb950", orange="#d29922",
    red="#f85149", purple="#bc8cff", teal="#39d353", pink="#f778ba",
    text="#e6edf3", dim="#8b949e",
    plot_bg="#161b22", grid="#21262d",
)
PAL = [T["blue"], T["green"], T["orange"], T["purple"],
       T["red"], "#79c0ff", "#56d364", "#ffa657", "#ff7b72", "#d2a8ff"]
FONT = "Segoe UI"

# apply dark style globally to matplotlib
plt.rcParams.update({
    "figure.facecolor": T["plot_bg"], "axes.facecolor": T["plot_bg"],
    "axes.edgecolor": T["border"], "axes.labelcolor": T["text"],
    "axes.titlecolor": T["text"], "axes.titlesize": 11,
    "axes.labelsize": 9, "axes.grid": True,
    "grid.color": T["grid"], "grid.linewidth": 0.5,
    "xtick.color": T["dim"], "ytick.color": T["dim"],
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.facecolor": T["card"], "legend.edgecolor": T["border"],
    "legend.fontsize": 8, "legend.labelcolor": T["text"],
    "text.color": T["text"], "savefig.facecolor": T["plot_bg"],
    "savefig.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════════════════
#  ML PIPELINE  —  pure functions, zero GUI coupling
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(file_path: str, target_column: str):
    """
    Load → engineer features → encode → scale → split.

    Returns
    -------
    X_tr, X_te, y_tr, y_te : DataFrames / Series
    meta : dict  — every stat the GUI needs, computed here from real data.
    """
    df_raw = pd.read_csv(file_path)
    missing_raw = int(df_raw.isnull().sum().sum())

    # drop pure-ID columns
    id_cols = [c for c in df_raw.columns if c.lower() in ("seqn", "id")]
    df = df_raw.drop(columns=id_cols, errors="ignore").copy()

    # ── feature engineering (only when source columns exist) ─────────────
    if "Baseline_FEV1_L" in df.columns and "Baseline_FVC_L" in df.columns:
        df["FEV1_FVC"] = (
            pd.to_numeric(df["Baseline_FEV1_L"], errors="coerce")
            / pd.to_numeric(df["Baseline_FVC_L"], errors="coerce").replace(0, np.nan)
        )
    if "Baseline_FEV1_L" in df.columns and "Age" in df.columns:
        df["Lung_Index"] = (
            pd.to_numeric(df["Baseline_FEV1_L"], errors="coerce")
            / pd.to_numeric(df["Age"], errors="coerce").replace(0, np.nan)
        )

    # engineer Obstruction using GOLD criterion if column absent
    if "Obstruction" not in df.columns:
        ratio = next(
            (c for c in ("Baseline_FEV1_FVC_Ratio", "FEV1_FVC")
             if c in df.columns), None)
        if ratio:
            df["Obstruction"] = (
                pd.to_numeric(df[ratio], errors="coerce") < 0.70
            ).astype(int)

    # validate target
    if target_column not in df.columns:
        raise ValueError(
            f"Target '{target_column}' not found.\n"
            f"Available columns:\n{df.columns.tolist()}")

    # ── impute ───────────────────────────────────────────────────────────
    for c in df.columns:
        if df[c].isnull().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                m = df[c].mode()
                if len(m):
                    df[c] = df[c].fillna(m.iloc[0])

    # ── X / y ────────────────────────────────────────────────────────────
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # detect task
    is_clf = (not pd.api.types.is_float_dtype(y)) or (y.nunique() <= 20)

    # encode target
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
    scaler = StandardScaler()
    feature_names = X.columns.tolist()
    X = pd.DataFrame(scaler.fit_transform(X),
                     columns=feature_names, index=X.index)

    # split
    strat = y if is_clf else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=strat)

    # numeric columns in raw df for distribution plots
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
        n_estimators=120, max_depth=6,
        random_state=SEED, n_jobs=-1)


# ── train + evaluate ─────────────────────────────────────────────────────────

def train_and_evaluate(model, X_tr, X_te, y_tr, y_te, is_clf, feat_names):
    """
    Fit → predict → compute every metric from live predictions.
    Returns a results dict with NO hardcoded values.
    """
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # feature importance — works identically for XGB and RF
    fi_arr = (model.feature_importances_
              if hasattr(model, "feature_importances_")
              else np.zeros(len(feat_names)))
    fi = pd.Series(fi_arr, index=feat_names).sort_values(ascending=False)

    res = dict(
        model=model, y_pred=y_pred, y_test=y_te,
        params=model.get_params(),
        model_class=type(model).__name__,
        feature_names=feat_names,
        feature_importance=fi,
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
#  PLOT HELPERS  —  pure functions that receive data, return Figure objects
# ══════════════════════════════════════════════════════════════════════════════

def _fig(nrows=1, ncols=1, figsize=(11, 5)):
    f, ax = plt.subplots(nrows, ncols, figsize=figsize)
    f.patch.set_facecolor(T["plot_bg"])
    for a in (np.array(ax).flatten() if hasattr(ax, "__iter__") else [ax]):
        a.set_facecolor(T["plot_bg"])
    return f, ax


def _embed(fig, parent):
    for w in parent.winfo_children():
        w.destroy()
    c = FigureCanvasTkAgg(fig, master=parent)
    c.draw()
    c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)


# ── TAB 1: Overview ──────────────────────────────────────────────────────────

def plot_overview(meta, rx=None, rr=None):
    """
    4-panel overview.  Every value from meta/results dicts.
    [0] target distribution  [1] scatter of first two relevant numerics
    [2] class balance / secondary hist  [3] pred vs actual (both models)
    """
    df   = meta["df_raw"]
    tgt  = meta["target_series"]
    fig, ax = _fig(2, 2, (13, 7))
    ax = ax.flatten()

    # [0] target distribution
    ax[0].hist(tgt.dropna(), bins=40, color=T["blue"], alpha=0.85,
               edgecolor=T["bg"], linewidth=0.3)
    med = tgt.median()
    ax[0].axvline(med, color=T["red"], ls="--", lw=1.2,
                  label=f"Median = {med:.3f}")
    ax[0].set_title(f"Target: {meta['target_column']}")
    ax[0].set_xlabel(meta["target_column"])
    ax[0].set_ylabel("Count")
    ax[0].legend()

    # [1] first two clinically relevant scatter cols detected at runtime
    xcol = next((c for c in ("Age",) if c in df.columns), None)
    ycol = next((c for c in ("Baseline_FEV1_L", "FEV1")
                 if c in df.columns), None)
    if xcol and ycol:
        sub = df[[xcol, ycol]].dropna()
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=SEED)
        ax[1].scatter(sub[xcol], sub[ycol], alpha=0.2, s=6, color=T["green"])
        ax[1].set_xlabel(xcol); ax[1].set_ylabel(ycol)
        ax[1].set_title(f"{xcol} vs {ycol}")
    else:
        ax[1].text(0.5, 0.5, "Age / FEV1 columns not found",
                   ha="center", va="center", transform=ax[1].transAxes,
                   color=T["dim"], fontsize=10)
        ax[1].set_title("Scatter (N/A)")

    # [2] class balance or BMI histogram
    if meta["is_clf"] and meta["class_dist"]:
        cls = list(meta["class_dist"].keys())
        cnt = list(meta["class_dist"].values())
        bars = ax[2].bar([str(c) for c in cls], cnt,
                         color=PAL[:len(cls)], width=0.5)
        for b, v in zip(bars, cnt):
            ax[2].text(b.get_x() + b.get_width()/2,
                       b.get_height() + max(cnt)*0.01,
                       f"{v:,}", ha="center", va="bottom",
                       fontsize=8, color=T["text"])
        ax[2].set_title("Class Distribution (Full Dataset)")
        ax[2].set_xlabel("Class"); ax[2].set_ylabel("Count")
    else:
        c2 = next((c for c in ("BMI","Weight","Height")
                    if c in df.columns), None)
        if c2:
            d = df[c2].dropna()
            ax[2].hist(d, bins=35, color=T["orange"], alpha=0.85,
                       edgecolor=T["bg"], linewidth=0.3)
            ax[2].axvline(d.median(), color=T["red"], ls="--", lw=1.2)
            ax[2].set_title(f"{c2} Distribution"); ax[2].set_xlabel(c2)
        ax[2].set_ylabel("Count")

    # [3] pred vs actual — overlays both models if present
    plotted = False
    for res, lbl, col in [(rx,"XGBoost",T["blue"]),(rr,"Random Forest",T["orange"])]:
        if res is None:
            continue
        yt = np.array(res["y_test"]); yp = np.array(res["y_pred"])
        if not meta["is_clf"]:
            ax[3].scatter(yt, yp, alpha=0.25, s=8, color=col, label=lbl)
        else:
            jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(yt))
            ax[3].scatter(yt+jit, yp+jit, alpha=0.15, s=6, color=col, label=lbl)
        plotted = True
    if plotted and not meta["is_clf"]:
        ref = rx or rr
        mn, mx = np.array(ref["y_test"]).min(), np.array(ref["y_test"]).max()
        ax[3].plot([mn,mx],[mn,mx],"r--",lw=1.2,label="Ideal")
    ax[3].set_title("Predicted vs Actual"); ax[3].set_xlabel("Actual")
    ax[3].set_ylabel("Predicted")
    if plotted:
        ax[3].legend(fontsize=7)

    fig.suptitle("Dataset Overview", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── TAB 2: Feature Importance ────────────────────────────────────────────────

def plot_feature_importance(rx=None, rr=None, top_n=20):
    models = [(r, n, c) for r, n, c in
              [(rx,"XGBoost",T["blue"]),(rr,"Random Forest",T["orange"])]
              if r is not None]
    ncols = len(models)
    fig, axes = _fig(1, ncols, (7*ncols, 8))
    if ncols == 1:
        axes = [axes]

    for ax, (res, name, color) in zip(axes, models):
        fi = res["feature_importance"].head(top_n)
        labels = [l[:35]+"…" if len(l)>35 else l for l in fi.index]
        bars = ax.barh(range(len(fi)), fi.values,
                       color=color, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{name} — Top {len(fi)} Features")
        for bar, val in zip(bars, fi.values):
            ax.text(val + fi.values.max()*0.008,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=7, color=T["text"])
    fig.tight_layout(pad=2.0)
    return fig


# ── TAB 3: Predictions ──────────────────────────────────────────────────────

def plot_predictions(rx=None, rr=None, is_clf=True):
    models = [(r,n,c) for r,n,c in
              [(rx,"XGBoost",T["blue"]),(rr,"Random Forest",T["orange"])]
              if r is not None]
    if is_clf:
        ncols = len(models)
        fig, axes = _fig(1, ncols, (6*ncols, 5))
        if ncols == 1: axes = [axes]
        for ax, (res, name, color) in zip(axes, models):
            cm = res["cm"]; n_cls = cm.shape[0]
            im = ax.imshow(cm, cmap="Blues", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if cm[i,j] > cm.max()/2 else T["text"])
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels([f"Pred {k}" for k in range(n_cls)], fontsize=8)
            ax.set_yticklabels([f"True {k}" for k in range(n_cls)], fontsize=8)
            roc = res.get("roc_auc")
            roc_str = f"  AUC={roc:.4f}" if roc is not None else ""
            ax.set_title(f"{name} — Confusion Matrix\n"
                         f"Accuracy={res['accuracy']:.4f}{roc_str}")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    else:
        fig, axes = _fig(len(models), 2, (12, 5*len(models)))
        if len(models) == 1: axes = np.array([axes])
        for row, (res, name, color) in enumerate(models):
            yt = np.array(res["y_test"]); yp = np.array(res["y_pred"])
            ax_s = axes[row][0]; ax_e = axes[row][1]
            ax_s.scatter(yt, yp, alpha=0.3, s=10, color=color)
            mn, mx = yt.min(), yt.max()
            ax_s.plot([mn,mx],[mn,mx],"r--",lw=1.2,label="Ideal")
            ax_s.set_title(f"{name} — Pred vs Actual  R²={res['r2']:.4f}")
            ax_s.set_xlabel("Actual"); ax_s.set_ylabel("Predicted")
            ax_s.legend(fontsize=7)
            errs = yp - yt
            ax_e.hist(errs, bins=40, color=color, alpha=0.8,
                      edgecolor=T["bg"], linewidth=0.3)
            ax_e.axvline(0, color=T["red"], ls="--", lw=1.2)
            ax_e.set_title(f"{name} — Error Dist.  MAE={res['mae']:.4f}")
            ax_e.set_xlabel("Error (Pred − Actual)"); ax_e.set_ylabel("Count")
    fig.tight_layout(pad=2.0)
    return fig


# ── TAB 4: Residuals / CM ───────────────────────────────────────────────────

def plot_residuals_cm(rx=None, rr=None, is_clf=True):
    models = [(r,n,c) for r,n,c in
              [(rx,"XGBoost",T["blue"]),(rr,"Random Forest",T["orange"])]
              if r is not None]
    if is_clf:
        fig, axes = _fig(len(models), 1, (9, 4*len(models)))
        if len(models) == 1: axes = [axes]
        for ax, (res, name, color) in zip(
                np.array(axes).flatten(), models):
            ax.axis("off")
            ax.text(0.02, 0.98, f"{name}  —  Classification Report",
                    transform=ax.transAxes, fontsize=11, color=color,
                    va="top", fontweight="bold")
            ax.text(0.02, 0.86, res["clf_report"],
                    transform=ax.transAxes, fontsize=9,
                    color=T["text"], va="top", fontfamily="monospace")
    else:
        fig, axes = _fig(len(models), 2, (12, 5*len(models)))
        if len(models) == 1: axes = np.array([axes])
        for row, (res, name, color) in enumerate(models):
            yt = np.array(res["y_test"]); yp = np.array(res["y_pred"])
            residuals = yp - yt
            # residual scatter
            ax_r = axes[row][0]
            ax_r.scatter(yp, residuals, alpha=0.3, s=8, color=color)
            ax_r.axhline(0, color=T["red"], ls="--", lw=1.2)
            ax_r.set_xlabel("Predicted"); ax_r.set_ylabel("Residual")
            ax_r.set_title(f"{name} — Residuals vs Predicted")
            # Q-Q plot via scipy
            ax_q = axes[row][1]
            (osm, osr), (slope, intercept, _) = sp_stats.probplot(residuals)
            ax_q.scatter(osm, osr, alpha=0.4, s=8, color=color)
            ax_q.plot(osm, slope*np.array(osm)+intercept,
                      color=T["red"], lw=1.5, label="Reference line")
            ax_q.set_xlabel("Theoretical Quantiles")
            ax_q.set_ylabel("Sample Quantiles")
            ax_q.set_title(f"{name} — Q-Q Plot")
            ax_q.legend(fontsize=7)
    fig.tight_layout(pad=2.0)
    return fig


# ── TAB 5: ROC / PR Curves ──────────────────────────────────────────────────

def plot_roc_pr(rx=None, rr=None):
    fig, (ax_r, ax_p) = _fig(1, 2, (12, 5))
    for res, name, color in [(rx,"XGBoost",T["blue"]),
                              (rr,"Random Forest",T["orange"])]:
        if res is None or res.get("roc_data") is None:
            continue
        fpr, tpr, _ = res["roc_data"]
        prec, rec, _ = res["pr_data"]
        ax_r.plot(fpr, tpr, color=color, lw=2,
                  label=f"{name}  AUC={res['roc_auc']:.4f}")
        ax_p.plot(rec, prec, color=color, lw=2,
                  label=f"{name}  AP={res['avg_precision']:.4f}")
    ax_r.plot([0,1],[0,1], color=T["dim"], ls="--", lw=1, label="Random")
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1.02)
    ax_r.set_xlabel("False Positive Rate")
    ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("ROC Curve"); ax_r.legend()
    ax_p.set_xlim(0,1); ax_p.set_ylim(0,1.02)
    ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision")
    ax_p.set_title("Precision-Recall Curve"); ax_p.legend()
    fig.tight_layout(pad=2.0)
    return fig


# ── TAB 6: Distributions ────────────────────────────────────────────────────

def plot_distributions(df_raw, numeric_cols, max_plots=9):
    """
    Histogram grid.  Column list is built dynamically:
      1) clinically relevant cols that exist in this specific CSV
      2) remaining numeric cols to fill up to max_plots
    """
    priority = [
        "Baseline_FEV1_L","Baseline_FVC_L","Baseline_FEV1_FVC_Ratio",
        "Baseline_PEF_Ls","Baseline_FEF2575_Ls",
        "FEV1_Zscores_GLOBAL","FVC_Zscores_GLOBAL",
        "Age","Height","Weight","BMI",
    ]
    ordered = ([c for c in priority if c in numeric_cols]
               + [c for c in numeric_cols if c not in priority])[:max_plots]
    n = len(ordered)
    if n == 0:
        fig, ax = _fig(1, 1, (6, 3))
        ax.text(0.5, 0.5, "No numeric columns found",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color=T["dim"])
        ax.axis("off")
        return fig

    ncols = 3; nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = _fig(nrows, ncols, (14, 4*nrows))
    flat = np.array(axes).flatten()

    for idx, col in enumerate(ordered):
        ax = flat[idx]
        data = pd.to_numeric(df_raw[col], errors="coerce").dropna()
        med = data.median()
        ax.hist(data, bins=40, color=PAL[idx % len(PAL)], alpha=0.85,
                edgecolor=T["bg"], linewidth=0.3)
        ax.axvline(med, color=T["red"], ls="--", lw=1.2,
                   label=f"Median={med:.3f}")
        short = col.replace("_"," ").replace("Baseline ","")
        ax.set_title(short[:28], fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Column Distributions (Live from Dataset)",
                 fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.0)
    return fig


# ── TAB 7: Model Comparison ─────────────────────────────────────────────────

def plot_model_comparison(rx, rr, is_clf):
    fig, (ax_b, ax_c) = _fig(1, 2, (13, 5))

    if is_clf:
        keys   = ["accuracy","roc_auc","avg_precision"]
        labels = ["Accuracy","ROC-AUC","Avg Precision"]
    else:
        keys   = ["mae","rmse","r2"]
        labels = ["MAE","RMSE","R²"]

    xv = [rx.get(k) or 0 for k in keys]
    rv = [rr.get(k) or 0 for k in keys]
    x = np.arange(len(keys)); w = 0.35

    b1 = ax_b.bar(x-w/2, xv, w, label="XGBoost",  color=T["blue"],  alpha=0.9)
    b2 = ax_b.bar(x+w/2, rv, w, label="Random Forest", color=T["orange"], alpha=0.9)
    for bars in [b1, b2]:
        for b in bars:
            ax_b.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                      f"{b.get_height():.4f}", ha="center", va="bottom",
                      fontsize=7, color=T["text"])
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Score"); ax_b.set_title("Metrics Comparison")
    ax_b.legend(); ax_b.set_ylim(0, max(max(xv), max(rv))*1.15)

    # right panel: ROC overlay (clf) or error dist overlay (reg)
    if is_clf:
        for res, n, c in [(rx,"XGBoost",T["blue"]),(rr,"Random Forest",T["orange"])]:
            if res.get("roc_data"):
                fpr, tpr, _ = res["roc_data"]
                ax_c.plot(fpr, tpr, color=c, lw=2,
                          label=f"{n} AUC={res['roc_auc']:.4f}")
        ax_c.plot([0,1],[0,1], color=T["dim"], ls="--", lw=1)
        ax_c.set_title("ROC Curves Overlaid")
        ax_c.set_xlabel("FPR"); ax_c.set_ylabel("TPR"); ax_c.legend()
    else:
        for res, n, c in [(rx,"XGBoost",T["blue"]),(rr,"Random Forest",T["orange"])]:
            errs = np.array(res["y_pred"]) - np.array(res["y_test"])
            ax_c.hist(errs, bins=35, alpha=0.5, color=c, label=n)
        ax_c.axvline(0, color=T["red"], ls="--", lw=1.2)
        ax_c.set_title("Error Distribution Comparison")
        ax_c.set_xlabel("Error"); ax_c.set_ylabel("Count"); ax_c.legend()

    fig.tight_layout(pad=2.0)
    return fig


# ── TAB 8: Conclusion text builder ──────────────────────────────────────────

def build_conclusion(meta, rx=None, rr=None):
    """Returns list of (text, tag) tuples. Every value from live dicts."""
    L = []
    def h(t):   L.append((t, "h"))
    def kv(k,v):L.append((f"  {k:<32}{v}", "kv"))
    def sep():  L.append(("─"*74, "sep"))
    def p(t):   L.append((t, "p"))
    def b():    L.append(("", "p"))

    h("MODEL TRAINING SUMMARY — Spirometer Smart Health Monitor")
    p(f"  Target: {meta['target_column']}  •  NHANES 2007-2012")
    sep()

    # dataset
    h("📂  DATASET")
    kv("File path:",         meta["file_path"])
    kv("Total rows:",        f"{meta['n_rows']:,}")
    kv("Total columns:",     f"{meta['n_cols']:,}")
    kv("Feature count:",     f"{meta['n_features']:,}")
    kv("Target column:",     meta["target_column"])
    kv("Task type:",         "Classification" if meta["is_clf"] else "Regression")
    kv("Train / Test:",      f"{meta['train_size']:,} / {meta['test_size']:,}  (80/20)")
    ts = meta["target_series"].dropna()
    kv("Target mean:",       f"{ts.mean():.4f}")
    kv("Target std:",        f"{ts.std():.4f}")
    kv("Target min / max:",  f"{ts.min():.4f} / {ts.max():.4f}")
    if meta["is_clf"] and meta["class_dist"]:
        kv("Class distribution:", str(meta["class_dist"]))
    kv("Missing values (raw):", f"{meta['missing_raw']:,}")
    b()

    def _model(res, label, tag):
        sep()
        L.append((f"{'🔵' if 'XG' in label else '🟠'}  {label}", tag))
        b()
        pr = res["params"]
        kv("Model class:",        res["model_class"])
        # show every hyperparameter that actually exists in this model
        for k in ("n_estimators","max_depth","learning_rate","subsample",
                  "colsample_bytree","class_weight","criterion","objective",
                  "random_state"):
            v = pr.get(k)
            if v is not None:
                kv(f"{k}:", str(v))
        b()
        L.append(("  📊  METRICS", tag))
        if meta["is_clf"]:
            acc = res["accuracy"]
            kv("Accuracy:",      f"{acc:.4f}  ({acc*100:.2f}%)")
            roc = res.get("roc_auc")
            kv("ROC-AUC:",       f"{roc:.4f}" if roc is not None else "N/A")
            ap  = res.get("avg_precision")
            kv("Avg Precision:", f"{ap:.4f}" if ap is not None else "N/A")
            b(); p("  Classification Report:"); L.append((res["clf_report"], "mono"))
        else:
            kv("MAE:",  f"{res['mae']:.4f}")
            kv("MSE:",  f"{res['mse']:.4f}")
            kv("RMSE:", f"{res['rmse']:.4f}")
            kv("R²:",   f"{res['r2']:.4f}")
        b()
        L.append(("  🏆  TOP 10 FEATURES", tag))
        for rank, (feat, score) in enumerate(
                res["feature_importance"].head(10).items(), 1):
            short = feat[:42]+"…" if len(feat)>42 else feat
            kv(f"  #{rank:>2}  {short}", f"{score:.4f}")
        b()

    if rx: _model(rx, "XGBOOST", "xgb")
    if rr: _model(rr, "RANDOM FOREST", "rf")

    # spirometry stats from raw data
    sep(); h("📈  KEY SPIROMETRY STATS (raw data)")
    stat_cols = [c for c in (
        "Baseline_FEV1_L","Baseline_FVC_L","Baseline_FEV1_FVC_Ratio",
        "Baseline_PEF_Ls","Baseline_FEF2575_Ls",
        "FEV1_Zscores_GLOBAL","FVC_Zscores_GLOBAL",
        "Age","Height","Weight","BMI",
    ) if c in meta["df_raw"].columns]
    for col in stat_cols:
        d = pd.to_numeric(meta["df_raw"][col], errors="coerce").dropna()
        kv(col, f"mean={d.mean():.3f}  std={d.std():.3f}  "
                f"min={d.min():.3f}  max={d.max():.3f}  n={len(d):,}")
    b()

    # comparison verdict
    if rx and rr:
        sep(); h("⚖️  COMPARISON VERDICT")
        if meta["is_clf"]:
            pairs = [("Best Accuracy:","accuracy",False),
                     ("Best ROC-AUC:","roc_auc",False),
                     ("Best Avg Precision:","avg_precision",False)]
        else:
            pairs = [("Best R²:","r2",False),
                     ("Lowest MAE:","mae",True),
                     ("Lowest RMSE:","rmse",True)]
        for label, key, lower in pairs:
            xv = rx.get(key) or 0; rv = rr.get(key) or 0
            if lower:
                winner = "XGBoost" if xv <= rv else "Random Forest"
            else:
                winner = "XGBoost" if xv >= rv else "Random Forest"
            tag = "xgb" if winner == "XGBoost" else "rf"
            L.append((f"  {label:<30}{winner}  ({xv:.4f} vs {rv:.4f})", tag))
        b()

    # clinical interpretation
    sep(); h("🩺  CLINICAL INTERPRETATION")
    b()
    p("  FEV1 (Forced Expiratory Volume in 1s) and FVC (Forced Vital Capacity)")
    p("  are primary spirometry markers. FEV1/FVC < 0.70 is the GOLD criterion")
    p("  for obstructive lung disease (asthma, COPD).")
    b()
    p("  Lung_Index = FEV1 / Age captures age-related decline.")
    p("  Height²×Race encodes population-specific reference equations.")
    b()
    for res, name in [(rx,"XGBoost"),(rr,"Random Forest")]:
        if res is None: continue
        tag = "xgb" if "XG" in name else "rf"
        if meta["is_clf"]:
            acc = res["accuracy"]; roc = res.get("roc_auc") or 0
            if acc > 0.95:
                grade = "Excellent"
            elif acc > 0.80:
                grade = "Good"
            else:
                grade = "Moderate"
            L.append((
                f"  ✅  {name}: {grade} — Accuracy {acc*100:.2f}%, "
                f"AUC {roc:.4f}", tag))
        else:
            r2 = res.get("r2",0)
            grade = "Excellent" if r2>0.85 else "Good" if r2>0.6 else "Moderate"
            L.append((
                f"  ✅  {name}: {grade} — R²={r2:.4f}, "
                f"explains {r2*100:.1f}% of variance", tag))
    b()
    return L


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Spirometer Monitor — XGBoost & Random Forest")
        self.configure(bg=T["bg"])
        self.minsize(1100, 680)
        try:
            self.state("zoomed")           # Windows maximize
        except tk.TclError:
            pass

        # state
        self._meta = None; self._rx = self._rr = None

        # vars
        self.v_path   = tk.StringVar(
            value=r"D:\New folder (2)\NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv")
        self.v_target = tk.StringVar(value="Obstruction")
        self.v_mode   = tk.StringVar(value="Compare Both")

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

    # ── top bar ──────────────────────────────────────────────────────────

    def _topbar(self):
        bar = tk.Frame(self, bg=T["sidebar"], height=44)
        bar.pack(fill=tk.X); bar.pack_propagate(False)
        tk.Label(bar, text="🫁", font=(FONT,14),
                 bg=T["sidebar"], fg=T["blue"]).pack(side=tk.LEFT, padx=(14,4))
        tk.Label(bar, text="Spirometer Smart Health Monitor",
                 font=(FONT,13,"bold"), bg=T["sidebar"],
                 fg=T["text"]).pack(side=tk.LEFT)
        self._lbl_sub = tk.Label(
            bar, text="XGBoost ↔ Random Forest  •  NHANES 2007–2012",
            font=(FONT,9), bg=T["sidebar"], fg=T["dim"])
        self._lbl_sub.pack(side=tk.LEFT, padx=16)
        self._lbl_dot = tk.Label(bar, text="● READY", font=(FONT,9,"bold"),
                                 bg=T["sidebar"], fg=T["green"])
        self._lbl_dot.pack(side=tk.RIGHT, padx=16)

    # ── sidebar (scrollable) ─────────────────────────────────────────────

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=T["sidebar"], width=234)
        sb.pack(side=tk.LEFT, fill=tk.Y); sb.pack_propagate(False)

        # canvas + scrollbar for the sidebar content
        canvas = tk.Canvas(sb, bg=T["sidebar"], highlightthickness=0, width=218)
        vsb = ttk.Scrollbar(sb, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=T["sidebar"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw", width=218)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # mousewheel
        def _mw(e):
            canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")
        canvas.bind_all("<MouseWheel>", _mw)

        def _sec(txt):
            tk.Frame(inner, bg=T["border"], height=1).pack(
                fill=tk.X, pady=(12,0))
            tk.Label(inner, text=txt, font=(FONT,8,"bold"),
                     bg=T["sidebar"], fg=T["blue"],
                     anchor="w").pack(fill=tk.X, padx=10, pady=(4,2))

        # DATASET
        _sec("▸  DATASET")
        tk.Label(inner, text="CSV file path:", font=(FONT,8),
                 bg=T["sidebar"], fg=T["dim"], anchor="w").pack(
                     fill=tk.X, padx=10)
        pf = tk.Frame(inner, bg=T["sidebar"])
        pf.pack(fill=tk.X, padx=10, pady=2)
        self._lbl_path = tk.Label(
            pf, textvariable=self.v_path, font=(FONT,7),
            bg=T["sidebar"], fg=T["text"], anchor="w",
            wraplength=170, justify="left", cursor="hand2")
        self._lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._lbl_path.bind("<Button-1>", self._browse)

        # TARGET
        _sec("◎  TARGET COLUMN")
        self._cb_tgt = ttk.Combobox(
            inner, textvariable=self.v_target,
            font=(FONT,9), state="normal", width=22)
        self._cb_tgt.pack(padx=10, pady=(2,6), fill=tk.X)

        # MODEL
        _sec("◈  MODEL")
        for lbl in ("XGBoost","Random Forest","Compare Both"):
            tk.Radiobutton(
                inner, text=lbl, variable=self.v_mode, value=lbl,
                font=(FONT,9), bg=T["sidebar"], fg=T["text"],
                selectcolor=T["border"], activebackground=T["sidebar"],
                activeforeground=T["blue"], indicatoron=True,
            ).pack(anchor="w", padx=14, pady=1)

        # PARAMS
        _sec("⚙  TRAINED PARAMS")
        self._pf = tk.Frame(inner, bg=T["sidebar"])
        self._pf.pack(fill=tk.X, padx=10, pady=4)
        self._show_params({})

        # RUN
        tk.Frame(inner, bg=T["border"], height=1).pack(fill=tk.X, pady=8)
        self._btn = tk.Button(
            inner, text="▶  Run", font=(FONT,11,"bold"),
            bg=T["green"], fg="#000", activebackground="#2ea043",
            relief="flat", cursor="hand2", pady=8,
            command=self._on_run)
        self._btn.pack(fill=tk.X, padx=10, pady=4)
        tk.Button(
            inner, text="💾  Save All Plots", font=(FONT,9),
            bg=T["card"], fg=T["text"], activebackground=T["border"],
            relief="flat", cursor="hand2", pady=5,
            command=self._save).pack(fill=tk.X, padx=10, pady=(0,8))

        # INFO
        _sec("ℹ  DATASET INFO")
        self._if = tk.Frame(inner, bg=T["sidebar"])
        self._if.pack(fill=tk.X, padx=10, pady=4)
        self._info = {}
        for k in ("Rows","Columns","Features","Task",
                  "Train Size","Test Size","Missing (raw)"):
            row = tk.Frame(self._if, bg=T["sidebar"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=k+":", font=(FONT,8),
                     bg=T["sidebar"], fg=T["dim"],
                     width=14, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="—", font=(FONT,8,"bold"),
                           bg=T["sidebar"], fg=T["text"], anchor="w")
            lbl.pack(side=tk.LEFT)
            self._info[k] = lbl

    # ── content area ─────────────────────────────────────────────────────

    def _content(self, parent):
        ct = tk.Frame(parent, bg=T["bg"])
        ct.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # metric card rows
        co = tk.Frame(ct, bg=T["bg"])
        co.pack(fill=tk.X, padx=8, pady=(6,0))
        self._crows = {}
        for mname, color in [("XGBoost",T["blue"]),
                              ("Random Forest",T["orange"])]:
            rf = tk.Frame(co, bg=T["bg"])
            rf.pack(fill=tk.X, pady=2)
            tk.Label(rf, text=mname, font=(FONT,8,"bold"),
                     bg=T["bg"], fg=color, width=14, anchor="e"
                     ).pack(side=tk.LEFT, padx=(0,6))
            cards = {}
            for met in ("Accuracy","ROC-AUC","Avg Precision",
                        "MAE","RMSE","R²"):
                cf = tk.Frame(rf, bg=T["card"],
                              highlightbackground=T["border"],
                              highlightthickness=1)
                cf.pack(side=tk.LEFT, padx=3, pady=2, ipadx=10, ipady=4)
                tk.Label(cf, text=met, font=(FONT,7),
                         bg=T["card"], fg=T["dim"]).pack()
                vl = tk.Label(cf, text="—", font=(FONT,13,"bold"),
                              bg=T["card"], fg=T["dim"])
                vl.pack()
                cards[met] = vl
            self._crows[mname] = cards

        # tabs
        sty = ttk.Style(); sty.theme_use("default")
        sty.configure("D.TNotebook", background=T["bg"], borderwidth=0)
        sty.configure("D.TNotebook.Tab", background=T["card"],
                       foreground=T["dim"], padding=[12,5],
                       font=(FONT,9))
        sty.map("D.TNotebook.Tab",
                background=[("selected",T["bg"])],
                foreground=[("selected",T["blue"])])

        self._nb = ttk.Notebook(ct, style="D.TNotebook")
        self._nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        tabs = [
            ("📈 Overview",        "_t_ov"),
            ("🏆 Feature Importance","_t_fi"),
            ("🎯 Predictions",     "_t_pr"),
            ("📉 Residuals / CM",  "_t_rc"),
            ("📡 ROC / PR Curves", "_t_rp"),
            ("📊 Distributions",   "_t_ds"),
            ("⚖️ Model Comparison","_t_mc"),
            ("📋 Conclusion",      "_t_cl"),
        ]
        for title, attr in tabs:
            f = tk.Frame(self._nb, bg=T["bg"])
            self._nb.add(f, text=title)
            setattr(self, attr, f)

        # conclusion scrollable text
        bar = ttk.Scrollbar(self._t_cl)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self._txt = tk.Text(
            self._t_cl, bg=T["card"], fg=T["text"],
            font=("Consolas",9), wrap=tk.WORD,
            yscrollcommand=bar.set, relief="flat",
            padx=14, pady=10, state=tk.DISABLED, cursor="arrow")
        self._txt.pack(fill=tk.BOTH, expand=True)
        bar.config(command=self._txt.yview)
        for tag, cfg in [
            ("h",    dict(font=("Consolas",11,"bold"), foreground=T["blue"])),
            ("kv",   dict(foreground=T["text"])),
            ("sep",  dict(foreground=T["border"])),
            ("p",    dict(foreground=T["dim"])),
            ("mono", dict(font=("Consolas",8), foreground=T["text"])),
            ("xgb",  dict(font=("Consolas",9,"bold"), foreground=T["blue"])),
            ("rf",   dict(font=("Consolas",9,"bold"), foreground=T["orange"])),
        ]:
            self._txt.tag_configure(tag, **cfg)

    # ──────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _browse(self, _=None):
        p = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if p:
            self.v_path.set(p)
            try:
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                self._cb_tgt["values"] = cols
            except Exception:
                pass

    def _dot(self, txt, fg=None):
        self._lbl_dot.config(text=txt, fg=fg or T["green"])
        self.update_idletasks()

    def _show_params(self, params):
        for w in self._pf.winfo_children():
            w.destroy()
        skip = {"num_class","verbosity","n_jobs","base_score","booster",
                "callbacks","early_stopping_rounds","enable_categorical",
                "feature_types","gamma","grow_policy","importance_type",
                "interaction_constraints","max_bin","max_cat_threshold",
                "max_cat_to_onehot","max_delta_step","max_leaves",
                "min_child_weight","missing","monotone_constraints",
                "multi_strategy","nthread","num_parallel_tree",
                "reg_alpha","reg_lambda","sampling_method",
                "scale_pos_weight","tree_method","validate_parameters",
                "device","min_samples_split","min_samples_leaf",
                "min_weight_fraction_leaf","max_features",
                "max_leaf_nodes","min_impurity_decrease","bootstrap",
                "oob_score","warm_start","ccp_alpha","max_samples",
                "verbose"}
        shown = {k:v for k,v in params.items()
                 if v is not None and k not in skip}
        for k, v in list(shown.items())[:12]:
            row = tk.Frame(self._pf, bg=T["sidebar"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{k}:", font=(FONT,7),
                     bg=T["sidebar"], fg=T["dim"],
                     width=17, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=str(v), font=(FONT,7,"bold"),
                     bg=T["sidebar"], fg=T["green"],
                     anchor="w").pack(side=tk.LEFT)

    def _upd_info(self, m):
        data = {
            "Rows":        f"{m['n_rows']:,}",
            "Columns":     f"{m['n_cols']:,}",
            "Features":    f"{m['n_features']:,}",
            "Task":        "Classification" if m["is_clf"] else "Regression",
            "Train Size":  f"{m['train_size']:,}",
            "Test Size":   f"{m['test_size']:,}",
            "Missing (raw)": f"{m['missing_raw']:,}",
        }
        for k, v in data.items():
            self._info[k].config(text=v,
                fg=T["orange"] if k=="Task" else T["text"])

    def _upd_cards(self, res, model_name, is_clf):
        cards = self._crows[model_name]
        if is_clf:
            mp = {"Accuracy":("accuracy",T["green"]),
                  "ROC-AUC":("roc_auc",T["blue"]),
                  "Avg Precision":("avg_precision",T["purple"]),
                  "MAE":(None,T["dim"]),
                  "RMSE":(None,T["dim"]),
                  "R²":(None,T["dim"])}
        else:
            mp = {"Accuracy":(None,T["dim"]),
                  "ROC-AUC":(None,T["dim"]),
                  "Avg Precision":(None,T["dim"]),
                  "MAE":("mae",T["orange"]),
                  "RMSE":("rmse",T["red"]),
                  "R²":("r2",T["green"])}
        for met, (key, color) in mp.items():
            vl = cards[met]
            if key and res.get(key) is not None:
                vl.config(text=f"{res[key]:.4f}", fg=color)
            else:
                vl.config(text="—", fg=T["dim"])

    def _reset_cards(self):
        for mn in self._crows:
            for met, vl in self._crows[mn].items():
                vl.config(text="—", fg=T["dim"])

    # ──────────────────────────────────────────────────────────────────────
    #  RUN
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

            rx = rr = None
            if mode in ("XGBoost","Compare Both"):
                self._dot("● TRAINING XGBoost", T["blue"])
                rx = train_and_evaluate(
                    _build_xgb(meta["is_clf"], n_cls),
                    X_tr, X_te, y_tr, y_te,
                    meta["is_clf"], meta["feature_names"])

            if mode in ("Random Forest","Compare Both"):
                self._dot("● TRAINING Random Forest", T["orange"])
                rr = train_and_evaluate(
                    _build_rf(meta["is_clf"]),
                    X_tr, X_te, y_tr, y_te,
                    meta["is_clf"], meta["feature_names"])

            self._rx = rx; self._rr = rr
            self.after(0, self._post_run)

        except Exception:
            err = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Pipeline Error", err))
            self.after(0, lambda: self._dot("● ERROR", T["red"]))
            self.after(0, lambda: self._btn.config(
                state=tk.NORMAL, text="▶  Run"))

    def _post_run(self):
        meta = self._meta; rx = self._rx; rr = self._rr
        is_clf = meta["is_clf"]

        self._upd_info(meta)
        last = rx or rr
        if last:
            self._show_params(last["params"])
        if rx:
            self._upd_cards(rx, "XGBoost", is_clf)
        if rr:
            self._upd_cards(rr, "Random Forest", is_clf)

        self._dot("● RENDERING", T["blue"])
        self._render(meta, rx, rr, is_clf)

        mode = self.v_mode.get()
        self._lbl_sub.config(
            text=f"{mode}  •  Target: {meta['target_column']}  •  NHANES 2007-2012")
        self._dot("● READY", T["green"])
        self._btn.config(state=tk.NORMAL, text="▶  Run")

    def _render(self, meta, rx, rr, is_clf):
        # overview
        _embed(plot_overview(meta, rx, rr), self._t_ov)
        # feature importance
        _embed(plot_feature_importance(rx, rr), self._t_fi)
        # predictions
        _embed(plot_predictions(rx, rr, is_clf), self._t_pr)
        # residuals / CM
        _embed(plot_residuals_cm(rx, rr, is_clf), self._t_rc)

        # ROC / PR
        if is_clf:
            _embed(plot_roc_pr(rx, rr), self._t_rp)
        else:
            for w in self._t_rp.winfo_children(): w.destroy()
            tk.Label(self._t_rp,
                     text="ROC / PR curves are only for classification tasks.",
                     font=(FONT,11), bg=T["bg"], fg=T["dim"]).pack(expand=True)

        # distributions
        _embed(plot_distributions(meta["df_raw"], meta["numeric_raw_cols"]),
               self._t_ds)

        # model comparison
        if rx and rr:
            _embed(plot_model_comparison(rx, rr, is_clf), self._t_mc)
        else:
            for w in self._t_mc.winfo_children(): w.destroy()
            tk.Label(self._t_mc,
                     text="Run 'Compare Both' to see comparison.",
                     font=(FONT,11), bg=T["bg"], fg=T["dim"]).pack(expand=True)

        # conclusion
        lines = build_conclusion(meta, rx, rr)
        t = self._txt; t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)
        for text, tag in lines:
            t.insert(tk.END, text+"\n", tag)
        t.config(state=tk.DISABLED)
        t.yview_moveto(0)

    # ──────────────────────────────────────────────────────────────────────
    #  SAVE
    # ──────────────────────────────────────────────────────────────────────

    def _save(self):
        if self._meta is None:
            messagebox.showinfo("No results","Train a model first."); return
        folder = filedialog.askdirectory(title="Select folder to save plots")
        if not folder: return

        meta = self._meta; rx = self._rx; rr = self._rr
        is_clf = meta["is_clf"]

        saves = {
            "overview.png":           plot_overview(meta, rx, rr),
            "feature_importance.png": plot_feature_importance(rx, rr),
            "predictions.png":        plot_predictions(rx, rr, is_clf),
            "residuals_cm.png":       plot_residuals_cm(rx, rr, is_clf),
            "distributions.png":      plot_distributions(
                                          meta["df_raw"], meta["numeric_raw_cols"]),
        }
        if is_clf:
            saves["roc_pr.png"] = plot_roc_pr(rx, rr)
        if rx and rr:
            saves["model_comparison.png"] = plot_model_comparison(rx, rr, is_clf)

        for fname, fig in saves.items():
            fig.savefig(os.path.join(folder, fname), bbox_inches="tight", dpi=150)
            plt.close(fig)

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
