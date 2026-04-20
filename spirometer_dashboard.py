"""
=============================================================================
  Spirometer Smart Health Monitoring System — XGBoost ML Dashboard
  Modern Tkinter GUI with full ML pipeline, plots, and conclusion panel
=============================================================================
"""

import warnings
import os
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.exceptions import DataConversionWarning
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─── Suppress noisy warnings ────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── Color Palette (dark modern theme) ──────────────────────────────────────
BG_DARK       = "#0d1117"
BG_CARD       = "#161b22"
BG_SIDEBAR    = "#0d1117"
ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_RED    = "#f85149"
ACCENT_PURPLE = "#bc8cff"
TEXT_PRIMARY  = "#e6edf3"
TEXT_SECONDARY= "#8b949e"
BORDER_COLOR  = "#30363d"
PLOT_BG       = "#161b22"
PLOT_GRID     = "#21262d"

CHART_COLORS  = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE,
                 ACCENT_PURPLE, ACCENT_RED, "#79c0ff", "#56d364"]

# ─── Default dataset & target ───────────────────────────────────────────────
DEFAULT_PATH   = r"D:\New folder (2)\NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"
DEFAULT_TARGET = "Baseline_FEV1_L"

# ─── Key spirometry columns for the dataset summary cards ───────────────────
SPIRO_KEY_COLS = [
    "Baseline_FEV1_L", "Baseline_FVC_L", "Baseline_FEV1_FVC_Ratio",
    "Baseline_PEF_Ls", "Baseline_FEF2575_Ls",
    "FEV1_Zscores_GLOBAL", "FVC_Zscores_GLOBAL",
    "Age", "Height", "Weight", "BMI",
]


# ════════════════════════════════════════════════════════════════════════════
#  ML PIPELINE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(file_path, target_column):
    """Load CSV, impute missing values, encode & scale features."""
    df = pd.read_csv(file_path)

    # Drop SEQN-like pure ID columns
    id_cols = [c for c in df.columns if c.lower() == "seqn"]
    df.drop(columns=id_cols, inplace=True, errors="ignore")

    # Impute
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols   = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Auto-detect task type
    if pd.api.types.is_numeric_dtype(y):
        is_clf = y.nunique() <= 20 and y.dtype in ["int64", "int32", "int16"]
    else:
        is_clf = True

    if is_clf and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), index=y.index)

    if categorical_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc = ohe.fit_transform(X[categorical_cols])
        enc_names = ohe.get_feature_names_out(categorical_cols)
        enc_df = pd.DataFrame(enc, columns=enc_names, index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), enc_df], axis=1)

    if numerical_cols:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    feature_names = X.columns.tolist()
    X.columns = [f"f{i}" for i in range(X.shape[1])]

    return df, X, y, is_clf, feature_names


def train_and_evaluate(X, y, is_clf):
    """Split, train XGBoost, evaluate and return all artefacts."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y if is_clf else None
    )

    if is_clf:
        n_cls = int(y_tr.nunique())
        model = xgb.XGBClassifier(
            objective="binary:logistic" if n_cls == 2 else "multi:softmax",
            num_class=n_cls if n_cls > 2 else None,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
            use_label_encoder=False,
        )
    else:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        )

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if is_clf:
        if hasattr(model, "predict_proba") and y_te.nunique() == 2:
            proba = model.predict_proba(X_te)[:, 1]
            y_pred = (proba > 0.5).astype(int)
        metrics = {
            "accuracy":             accuracy_score(y_te, y_pred),
            "confusion_matrix":     confusion_matrix(y_te, y_pred),
            "classification_report":classification_report(y_te, y_pred, zero_division=0),
        }
    else:
        mae  = mean_absolute_error(y_te, y_pred)
        mse  = mean_squared_error(y_te, y_pred)
        metrics = {
            "mae":   mae,
            "mse":   mse,
            "rmse":  np.sqrt(mse),
            "r2":    r2_score(y_te, y_pred),
        }

    return model, X_tr, X_te, y_tr, y_te, y_pred, metrics


# ════════════════════════════════════════════════════════════════════════════
#  HELPER — styled matplotlib figure
# ════════════════════════════════════════════════════════════════════════════

def styled_fig(nrows=1, ncols=1, figsize=(8, 5), title=""):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=PLOT_BG)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax in axes.flat:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(TEXT_SECONDARY)
        ax.yaxis.label.set_color(TEXT_SECONDARY)
        ax.title.set_color(TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)
        ax.grid(True, color=PLOT_GRID, linewidth=0.5, linestyle="--")
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ════════════════════════════════════════════════════════════════════════════

class SpiroDashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("🫁  Spirometer Smart Health Monitor — XGBoost ML Dashboard")
        self.geometry("1380x860")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # State
        self.dataset_path  = tk.StringVar(value=DEFAULT_PATH)
        self.target_col    = tk.StringVar(value=DEFAULT_TARGET)
        self.status_text   = tk.StringVar(value="Ready — load dataset and press  ▶ Train Model")
        self.df            = None
        self.model         = None
        self.metrics       = None
        self.feature_names = None
        self.is_clf        = None
        self.X_te = self.y_te = self.y_pred = None

        self._build_ui()

    # ── Build full UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top Header bar
        self._header()
        # ── Main content: sidebar + notebook
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self._sidebar(body)

        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._metric_cards(right)
        self._tab_notebook(right)
        self._status_bar()

    # ── Header ────────────────────────────────────────────────────────────

    def _header(self):
        hdr = tk.Frame(self, bg=BG_CARD, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🫁  Spirometer Smart Health Monitor",
                 bg=BG_CARD, fg=ACCENT_BLUE,
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=20, pady=10)

        tk.Label(hdr, text="XGBoost ML Pipeline  •  NHANES 2007–2012",
                 bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Segoe UI", 10)).pack(side="left", padx=4)

        # Right-side badge
        tk.Label(hdr, text="● LIVE", bg=BG_CARD, fg=ACCENT_GREEN,
                 font=("Segoe UI", 9, "bold")).pack(side="right", padx=20)

    # ── Sidebar (controls) ───────────────────────────────────────────────

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG_CARD, width=270)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        def section(label):
            tk.Label(sb, text=label, bg=BG_CARD, fg=ACCENT_BLUE,
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(16, 2))
            tk.Frame(sb, bg=BORDER_COLOR, height=1).pack(fill="x", padx=14)

        # ── Dataset
        section("📂  DATASET")
        tk.Label(sb, text="CSV Path:", bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(8, 2))
        path_row = tk.Frame(sb, bg=BG_CARD)
        path_row.pack(fill="x", padx=14)
        tk.Entry(path_row, textvariable=self.dataset_path, bg="#21262d",
                 fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                 relief="flat", font=("Consolas", 8), width=24).pack(side="left", fill="x", expand=True)
        tk.Button(path_row, text="…", bg=BG_DARK, fg=TEXT_PRIMARY,
                  relief="flat", font=("Segoe UI", 9),
                  command=self._browse_file).pack(side="left", padx=(4, 0))

        # ── Target column
        section("🎯  TARGET COLUMN")
        tk.Label(sb, text="Column name:", bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(8, 2))
        self.target_entry = tk.Entry(sb, textvariable=self.target_col,
                                     bg="#21262d", fg=ACCENT_ORANGE,
                                     insertbackground=ACCENT_ORANGE,
                                     relief="flat", font=("Consolas", 9))
        self.target_entry.pack(fill="x", padx=14)

        # ── Model params (read-only info)
        section("⚙️  MODEL PARAMETERS")
        params = [
            ("n_estimators",   "200"),
            ("max_depth",      "5"),
            ("learning_rate",  "0.05"),
            ("subsample",      "0.8"),
            ("colsample_bytree","0.8"),
            ("random_state",   "42"),
            ("test_size",      "20 %"),
        ]
        for k, v in params:
            row = tk.Frame(sb, bg=BG_CARD)
            row.pack(fill="x", padx=14, pady=1)
            tk.Label(row, text=k, bg=BG_CARD, fg=TEXT_SECONDARY,
                     font=("Consolas", 8), width=18, anchor="w").pack(side="left")
            tk.Label(row, text=v, bg=BG_CARD, fg=ACCENT_GREEN,
                     font=("Consolas", 8, "bold")).pack(side="left")

        # ── Train button
        tk.Frame(sb, bg=BORDER_COLOR, height=1).pack(fill="x", padx=14, pady=14)
        self.train_btn = tk.Button(
            sb, text="▶  Train Model", bg=ACCENT_BLUE, fg=BG_DARK,
            font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
            activebackground="#79c0ff", activeforeground=BG_DARK,
            command=self._run_pipeline_thread
        )
        self.train_btn.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            sb, text="💾  Save All Plots", bg=BG_DARK, fg=TEXT_PRIMARY,
            font=("Segoe UI", 9), relief="flat", cursor="hand2",
            activebackground=BORDER_COLOR,
            command=self._save_plots
        ).pack(fill="x", padx=14, pady=(0, 16))

        # ── Dataset stats (populated after load)
        section("📊  DATASET INFO")
        self.info_frame = tk.Frame(sb, bg=BG_CARD)
        self.info_frame.pack(fill="x", padx=14, pady=6)
        self._info_lbl("Rows",     "—")
        self._info_lbl("Columns",  "—")
        self._info_lbl("Task",     "—")
        self._info_lbl("Features", "—")

    def _info_lbl(self, key, val):
        row = tk.Frame(self.info_frame, bg=BG_CARD)
        row.pack(fill="x", pady=1)
        lbl = tk.Label(row, text=key + ":", bg=BG_CARD, fg=TEXT_SECONDARY,
                       font=("Segoe UI", 8), width=10, anchor="w")
        lbl.pack(side="left")
        val_lbl = tk.Label(row, text=val, bg=BG_CARD, fg=TEXT_PRIMARY,
                           font=("Consolas", 8, "bold"))
        val_lbl.pack(side="left")
        # Use a safe attribute key (lowercase, spaces→underscores)
        safe_key = key.lower().replace(" ", "_")
        setattr(self, f"_info_{safe_key}", val_lbl)

    # ── Metric Cards ─────────────────────────────────────────────────────

    def _metric_cards(self, parent):
        self.cards_frame = tk.Frame(parent, bg=BG_DARK)
        self.cards_frame.pack(fill="x", pady=(6, 0))
        self.card_widgets       = {}
        self.card_label_widgets = {}
        placeholders = [
            ("MAE",  "—", ACCENT_ORANGE),
            ("RMSE", "—", ACCENT_RED),
            ("R²",   "—", ACCENT_BLUE),
            ("MSE",  "—", ACCENT_PURPLE),
            ("Train Size", "—", ACCENT_GREEN),
            ("Test Size",  "—", TEXT_SECONDARY),
        ]
        for label, val, color in placeholders:
            c = self._make_card(self.cards_frame, label, val, color)
            c.pack(side="left", padx=5, pady=4, fill="both", expand=True)

    def _make_card(self, parent, label, value, color):
        card = tk.Frame(parent, bg=BG_CARD, relief="flat",
                        highlightbackground=BORDER_COLOR,
                        highlightthickness=1)
        # Store header label so we can rename it (e.g. R² → Accuracy)
        hdr_lbl = tk.Label(card, text=label, bg=BG_CARD, fg=TEXT_SECONDARY,
                           font=("Segoe UI", 8))
        hdr_lbl.pack(pady=(8, 0))
        val_lbl = tk.Label(card, text=value, bg=BG_CARD, fg=color,
                           font=("Segoe UI", 16, "bold"))
        val_lbl.pack(pady=(0, 8))
        self.card_widgets[label]              = val_lbl
        self.card_label_widgets[label]        = hdr_lbl
        return card

    def _update_cards(self, metrics, train_size, test_size, is_clf):
        if is_clf:
            self.card_widgets["MAE"].config(text="N/A")
            self.card_widgets["RMSE"].config(text="N/A")
            self.card_widgets["MSE"].config(text="N/A")
            acc = metrics["accuracy"]
            self.card_widgets["R²"].config(
                text=f"{acc:.4f}",
                fg=ACCENT_GREEN if acc > 0.75 else ACCENT_RED
            )
            # Relabel the header of the R² card to "Accuracy" for classification
            self.card_label_widgets["R²"].config(text="Accuracy")
        else:
            self.card_widgets["MAE"].config(text=f"{metrics['mae']:.4f}")
            self.card_widgets["RMSE"].config(
                text=f"{metrics['rmse']:.4f}",
                fg=ACCENT_GREEN if metrics['r2'] > 0.6 else ACCENT_RED
            )
            r2_color = ACCENT_GREEN if metrics['r2'] > 0.6 else ACCENT_RED
            self.card_widgets["R²"].config(
                text=f"{metrics['r2']:.4f}", fg=r2_color
            )
            self.card_label_widgets["R²"].config(text="R²")
            self.card_widgets["MSE"].config(text=f"{metrics['mse']:.4f}")
        self.card_widgets["Train Size"].config(text=f"{train_size:,}")
        self.card_widgets["Test Size"].config(text=f"{test_size:,}")

    # ── Tab notebook ─────────────────────────────────────────────────────

    def _tab_notebook(self, parent):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=BG_CARD, foreground=TEXT_SECONDARY,
                        padding=[14, 6], font=("Segoe UI", 9))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_DARK)],
                  foreground=[("selected", ACCENT_BLUE)])

        self.nb = ttk.Notebook(parent, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True, pady=(8, 0))

        self.tab_overview    = self._make_tab("📈  Overview")
        self.tab_importance  = self._make_tab("🏆  Feature Importance")
        self.tab_pred        = self._make_tab("🎯  Predictions")
        self.tab_residuals   = self._make_tab("📉  Residuals / CM")
        self.tab_dist        = self._make_tab("📊  Distributions")
        self.tab_conclusion  = self._make_tab("📋  Conclusion")

    def _make_tab(self, label):
        frame = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(frame, text=label)
        return frame

    # ── Status bar ───────────────────────────────────────────────────────

    def _status_bar(self):
        bar = tk.Frame(self, bg=BG_CARD, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.status_text,
                 bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Segoe UI", 8)).pack(side="left", padx=12, pady=4)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=12, pady=6)

    # ── File browser ─────────────────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.dataset_path.set(path)

    # ════════════════════════════════════════════════════════════════════
    #  PIPELINE EXECUTION (background thread)
    # ════════════════════════════════════════════════════════════════════

    def _run_pipeline_thread(self):
        self.train_btn.config(state="disabled", text="⏳  Training…")
        self.progress.start(10)
        self.status_text.set("Running ML pipeline…")
        t = threading.Thread(target=self._run_pipeline, daemon=True)
        t.start()

    def _run_pipeline(self):
        try:
            path   = self.dataset_path.get()
            target = self.target_col.get().strip()

            self.status_text.set("Loading & preprocessing dataset…")
            df, X, y, is_clf, feature_names = load_and_preprocess(path, target)

            self.status_text.set("Training XGBoost model…")
            model, X_tr, X_te, y_tr, y_te, y_pred, metrics = train_and_evaluate(X, y, is_clf)

            # Store state
            self.df            = df
            self.model         = model
            self.metrics       = metrics
            self.feature_names = feature_names
            self.is_clf        = is_clf
            self.X_te  = X_te
            self.y_te  = y_te
            self.y_pred = y_pred
            self._X_tr = X_tr
            self._y_tr = y_tr

            self.status_text.set("Building dashboard plots…")
            self.after(0, self._populate_dashboard,
                       df, model, X_te, y_te, y_pred,
                       metrics, is_clf, feature_names,
                       len(X_tr), len(X_te))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Pipeline Error", str(e)))
            self.after(0, self._reset_btn)

    def _reset_btn(self):
        self.train_btn.config(state="normal", text="▶  Train Model")
        self.progress.stop()

    # ════════════════════════════════════════════════════════════════════
    #  POPULATE ALL TABS
    # ════════════════════════════════════════════════════════════════════

    def _populate_dashboard(self, df, model, X_te, y_te, y_pred,
                            metrics, is_clf, feature_names,
                            train_size, test_size):
        self._update_info_panel(df, is_clf, len(feature_names))
        self._update_cards(metrics, train_size, test_size, is_clf)

        self._tab_overview_plot(df, y_te, y_pred, is_clf, metrics)
        self._tab_importance_plot(model, feature_names)
        self._tab_pred_plot(y_te, y_pred, is_clf)
        self._tab_residuals_plot(y_te, y_pred, is_clf, metrics)
        self._tab_dist_plot(df)
        self._tab_conclusion(df, metrics, is_clf, feature_names, model,
                             train_size, test_size)

        self.status_text.set("✅  Pipeline complete — all plots rendered.")
        self._reset_btn()

    # ── Update sidebar info ───────────────────────────────────────────

    def _update_info_panel(self, df, is_clf, n_features):
        self._info_rows.config(text=f"{df.shape[0]:,}")
        self._info_columns.config(text=str(df.shape[1]))
        self._info_task.config(
            text="Classification" if is_clf else "Regression",
            fg=ACCENT_ORANGE if is_clf else ACCENT_BLUE
        )
        self._info_features.config(text=str(n_features))

    # ── Embed matplotlib figure in a tab ─────────────────────────────

    def _embed(self, fig, tab):
        for w in tab.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ════════════════════════════════════════════════════════════════════
    #  TAB 1 — Overview (4-panel summary)
    # ════════════════════════════════════════════════════════════════════

    def _tab_overview_plot(self, df, y_te, y_pred, is_clf, metrics):
        fig = Figure(figsize=(12, 6), facecolor=PLOT_BG)
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.45, wspace=0.35)

        def ax_style(ax, title):
            ax.set_facecolor(PLOT_BG)
            ax.set_title(title, color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
            ax.xaxis.label.set_color(TEXT_SECONDARY)
            ax.yaxis.label.set_color(TEXT_SECONDARY)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER_COLOR)
            ax.grid(True, color=PLOT_GRID, linewidth=0.4, linestyle="--")

        # 1. FEV1 distribution
        ax1 = fig.add_subplot(gs[0, 0])
        col = "Baseline_FEV1_L"
        if col in df.columns:
            data = df[col].dropna()
            ax1.hist(data, bins=40, color=ACCENT_BLUE, alpha=0.85, edgecolor="none")
        ax_style(ax1, "FEV1 Distribution (L)")
        ax1.set_xlabel("FEV1 (L)")
        ax1.set_ylabel("Count")

        # 2. FVC distribution
        ax2 = fig.add_subplot(gs[0, 1])
        col2 = "Baseline_FVC_L"
        if col2 in df.columns:
            data2 = df[col2].dropna()
            ax2.hist(data2, bins=40, color=ACCENT_GREEN, alpha=0.85, edgecolor="none")
        ax_style(ax2, "FVC Distribution (L)")
        ax2.set_xlabel("FVC (L)")

        # 3. FEV1/FVC ratio distribution
        ax3 = fig.add_subplot(gs[0, 2])
        col3 = "Baseline_FEV1_FVC_Ratio"
        if col3 in df.columns:
            data3 = pd.to_numeric(df[col3], errors="coerce").dropna()
            ax3.hist(data3, bins=40, color=ACCENT_ORANGE, alpha=0.85, edgecolor="none")
        ax_style(ax3, "FEV1/FVC Ratio Distribution")
        ax3.set_xlabel("FEV1/FVC")

        # 4. Age vs FEV1 scatter
        ax4 = fig.add_subplot(gs[1, 0])
        if "Age" in df.columns and "Baseline_FEV1_L" in df.columns:
            ax4.scatter(df["Age"], df["Baseline_FEV1_L"],
                        alpha=0.2, s=6, color=ACCENT_PURPLE)
        ax_style(ax4, "Age vs FEV1")
        ax4.set_xlabel("Age (years)")
        ax4.set_ylabel("FEV1 (L)")

        # 5. BMI vs FVC
        ax5 = fig.add_subplot(gs[1, 1])
        if "BMI" in df.columns and "Baseline_FVC_L" in df.columns:
            ax5.scatter(df["BMI"], df["Baseline_FVC_L"],
                        alpha=0.2, s=6, color=ACCENT_RED)
        ax_style(ax5, "BMI vs FVC")
        ax5.set_xlabel("BMI")
        ax5.set_ylabel("FVC (L)")

        # 6. Predicted vs Actual (quick view)
        ax6 = fig.add_subplot(gs[1, 2])
        if not is_clf:
            ax6.scatter(y_te, y_pred, alpha=0.3, s=6, color=ACCENT_BLUE)
            mn, mx = float(y_te.min()), float(y_te.max())
            ax6.plot([mn, mx], [mn, mx], color=ACCENT_RED,
                     linestyle="--", linewidth=1.5, label="Perfect fit")
            ax6.legend(fontsize=7, labelcolor=TEXT_SECONDARY,
                       facecolor=BG_CARD, edgecolor=BORDER_COLOR)
        else:
            cm = metrics["confusion_matrix"]
            ax6.matshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax6.text(j, i, str(v), ha="center", va="center",
                         color=TEXT_PRIMARY, fontsize=8)
        ax_style(ax6, "Pred vs Actual" if not is_clf else "Confusion Matrix")
        if not is_clf:
            ax6.set_xlabel("Actual")
            ax6.set_ylabel("Predicted")

        self._embed(fig, self.tab_overview)

    # ════════════════════════════════════════════════════════════════════
    #  TAB 2 — Feature Importance
    # ════════════════════════════════════════════════════════════════════

    def _tab_importance_plot(self, model, feature_names):
        scores = model.get_booster().get_score(importance_type="weight")
        mapped = {feature_names[int(k.replace("f", ""))]: v
                  for k, v in scores.items()}
        top20  = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:20]
        names  = [x[0] for x in top20][::-1]
        vals   = [x[1] for x in top20][::-1]

        fig = Figure(figsize=(12, 7), facecolor=PLOT_BG)
        ax  = fig.add_subplot(111, facecolor=PLOT_BG)

        colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(names))]
        bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{int(val)}", va="center", color=TEXT_SECONDARY, fontsize=7.5)

        ax.set_title("Top 20 Feature Importance (F-Score / Weight)",
                     color=TEXT_PRIMARY, fontsize=11, fontweight="bold")
        ax.set_xlabel("F-Score (Weight)", color=TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_COLOR)
        ax.grid(True, axis="x", color=PLOT_GRID, linewidth=0.5, linestyle="--")
        fig.tight_layout()
        self._embed(fig, self.tab_importance)

    # ════════════════════════════════════════════════════════════════════
    #  TAB 3 — Predictions
    # ════════════════════════════════════════════════════════════════════

    def _tab_pred_plot(self, y_te, y_pred, is_clf):
        fig = Figure(figsize=(12, 6), facecolor=PLOT_BG)

        if not is_clf:
            gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
            # Scatter
            ax1 = fig.add_subplot(gs[0, 0], facecolor=PLOT_BG)
            ax1.scatter(y_te, y_pred, alpha=0.35, s=8, color=ACCENT_BLUE, label="Samples")
            mn, mx = float(y_te.min()), float(y_te.max())
            ax1.plot([mn, mx], [mn, mx], color=ACCENT_RED, linestyle="--",
                     linewidth=1.5, label="Perfect fit")
            ax1.set_title("Predicted vs Actual", color=TEXT_PRIMARY,
                          fontsize=10, fontweight="bold")
            ax1.set_xlabel("Actual Values")
            ax1.set_ylabel("Predicted Values")
            ax1.legend(fontsize=8, facecolor=BG_CARD,
                       labelcolor=TEXT_SECONDARY, edgecolor=BORDER_COLOR)

            # Error histogram
            ax2 = fig.add_subplot(gs[0, 1], facecolor=PLOT_BG)
            errors = np.array(y_pred) - np.array(y_te)
            ax2.hist(errors, bins=50, color=ACCENT_ORANGE, alpha=0.85, edgecolor="none")
            ax2.axvline(0, color=ACCENT_RED, linestyle="--", linewidth=1.5)
            ax2.set_title("Prediction Error Distribution", color=TEXT_PRIMARY,
                          fontsize=10, fontweight="bold")
            ax2.set_xlabel("Error (Predicted − Actual)")
            ax2.set_ylabel("Count")

            for ax in [ax1, ax2]:
                ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
                ax.xaxis.label.set_color(TEXT_SECONDARY)
                ax.yaxis.label.set_color(TEXT_SECONDARY)
                for sp in ax.spines.values():
                    sp.set_edgecolor(BORDER_COLOR)
                ax.grid(True, color=PLOT_GRID, linewidth=0.4, linestyle="--")
        else:
            # Class distribution
            ax = fig.add_subplot(111, facecolor=PLOT_BG)
            classes, counts = np.unique(y_pred, return_counts=True)
            ax.bar([str(c) for c in classes], counts,
                   color=CHART_COLORS[:len(classes)], edgecolor="none")
            ax.set_title("Predicted Class Distribution",
                         color=TEXT_PRIMARY, fontsize=10, fontweight="bold")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER_COLOR)
            ax.grid(True, axis="y", color=PLOT_GRID, linewidth=0.4)

        fig.tight_layout()
        self._embed(fig, self.tab_pred)

    # ════════════════════════════════════════════════════════════════════
    #  TAB 4 — Residuals / Confusion Matrix
    # ════════════════════════════════════════════════════════════════════

    def _tab_residuals_plot(self, y_te, y_pred, is_clf, metrics):
        fig = Figure(figsize=(12, 6), facecolor=PLOT_BG)

        if not is_clf:
            gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
            residuals = np.array(y_te) - np.array(y_pred)

            # Residuals vs Predicted
            ax1 = fig.add_subplot(gs[0, 0], facecolor=PLOT_BG)
            ax1.scatter(y_pred, residuals, alpha=0.35, s=8, color=ACCENT_PURPLE)
            ax1.axhline(0, color=ACCENT_RED, linestyle="--", linewidth=1.5)
            ax1.set_title("Residuals vs Predicted", color=TEXT_PRIMARY,
                          fontsize=10, fontweight="bold")
            ax1.set_xlabel("Predicted Values")
            ax1.set_ylabel("Residuals")

            # QQ-like: sorted residuals
            ax2 = fig.add_subplot(gs[0, 1], facecolor=PLOT_BG)
            sorted_res = np.sort(residuals)
            n = len(sorted_res)
            theoretical = np.linspace(-3, 3, n)
            ax2.scatter(theoretical, sorted_res, alpha=0.35, s=6, color=ACCENT_GREEN)
            ax2.plot([-3, 3], [-3*np.std(sorted_res), 3*np.std(sorted_res)],
                     color=ACCENT_RED, linestyle="--", linewidth=1.5)
            ax2.set_title("Residuals Q-Q Plot", color=TEXT_PRIMARY,
                          fontsize=10, fontweight="bold")
            ax2.set_xlabel("Theoretical Quantiles")
            ax2.set_ylabel("Sample Quantiles")

            for ax in [ax1, ax2]:
                ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
                ax.xaxis.label.set_color(TEXT_SECONDARY)
                ax.yaxis.label.set_color(TEXT_SECONDARY)
                for sp in ax.spines.values():
                    sp.set_edgecolor(BORDER_COLOR)
                ax.grid(True, color=PLOT_GRID, linewidth=0.4, linestyle="--")
        else:
            # Full styled confusion matrix
            cm = metrics["confusion_matrix"]
            ax = fig.add_subplot(111, facecolor=PLOT_BG)
            cax = ax.matshow(cm, cmap="Blues")
            fig.colorbar(cax)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center",
                        color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
            ax.set_title("Confusion Matrix", color=TEXT_PRIMARY,
                         fontsize=10, fontweight="bold", pad=20)
            ax.set_xlabel("Predicted Label", color=TEXT_SECONDARY)
            ax.set_ylabel("True Label", color=TEXT_SECONDARY)
            ax.tick_params(colors=TEXT_SECONDARY)

        fig.tight_layout()
        self._embed(fig, self.tab_residuals)

    # ════════════════════════════════════════════════════════════════════
    #  TAB 5 — Distributions (key spirometry parameters)
    # ════════════════════════════════════════════════════════════════════

    def _tab_dist_plot(self, df):
        key_cols = [c for c in SPIRO_KEY_COLS if c in df.columns][:9]
        n = len(key_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig = Figure(figsize=(13, 4 * nrows), facecolor=PLOT_BG)
        gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                                hspace=0.55, wspace=0.35)

        for idx, col in enumerate(key_cols):
            r, c = divmod(idx, ncols)
            ax = fig.add_subplot(gs[r, c], facecolor=PLOT_BG)
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            color = CHART_COLORS[idx % len(CHART_COLORS)]
            ax.hist(data, bins=35, color=color, alpha=0.85, edgecolor="none")

            # Median line
            med = data.median()
            ax.axvline(med, color=ACCENT_RED, linestyle="--",
                       linewidth=1.2, label=f"Median: {med:.2f}")
            ax.legend(fontsize=6.5, facecolor=BG_CARD,
                      labelcolor=TEXT_SECONDARY, edgecolor=BORDER_COLOR)

            short = col.replace("Baseline_", "").replace("_", " ")
            ax.set_title(short, color=TEXT_PRIMARY, fontsize=8, fontweight="bold")
            ax.set_ylabel("Count", fontsize=7)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
            ax.xaxis.label.set_color(TEXT_SECONDARY)
            ax.yaxis.label.set_color(TEXT_SECONDARY)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER_COLOR)
            ax.grid(True, color=PLOT_GRID, linewidth=0.4, linestyle="--")

        fig.tight_layout()
        self._embed(fig, self.tab_dist)

    # ════════════════════════════════════════════════════════════════════
    #  TAB 6 — Conclusion Panel
    # ════════════════════════════════════════════════════════════════════

    def _tab_conclusion(self, df, metrics, is_clf, feature_names,
                        model, train_size, test_size):
        for w in self.tab_conclusion.winfo_children():
            w.destroy()

        # Scrollable text widget
        frame = tk.Frame(self.tab_conclusion, bg=BG_DARK)
        frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        txt = tk.Text(frame, bg=BG_CARD, fg=TEXT_PRIMARY,
                      font=("Consolas", 9), relief="flat",
                      padx=20, pady=16, wrap="word",
                      yscrollcommand=scrollbar.set)
        txt.pack(fill="both", expand=True)
        scrollbar.config(command=txt.yview)

        # Tag styles
        txt.tag_configure("title",   foreground=ACCENT_BLUE,   font=("Consolas", 13, "bold"))
        txt.tag_configure("section", foreground=ACCENT_ORANGE, font=("Consolas", 10, "bold"))
        txt.tag_configure("key",     foreground=ACCENT_GREEN,  font=("Consolas", 9, "bold"))
        txt.tag_configure("value",   foreground=TEXT_PRIMARY,  font=("Consolas", 9))
        txt.tag_configure("note",    foreground=TEXT_SECONDARY,font=("Consolas", 8, "italic"))
        txt.tag_configure("good",    foreground=ACCENT_GREEN,  font=("Consolas", 9, "bold"))
        txt.tag_configure("bad",     foreground=ACCENT_RED,    font=("Consolas", 9, "bold"))
        txt.tag_configure("sep",     foreground=BORDER_COLOR,  font=("Consolas", 9))

        def line(text="", tag="value"):
            txt.insert("end", text + "\n", tag)

        def kv(k, v, vtag="value"):
            txt.insert("end", f"  {k:<28}", "key")
            txt.insert("end", f"{v}\n", vtag)

        # ── Header
        line("=" * 68, "sep")
        line("  MODEL TRAINING SUMMARY — Spirometer Smart Health Monitor", "title")
        line("  XGBoost ML Pipeline  •  NHANES 2007–2012", "note")
        line("=" * 68, "sep")
        line()

        # ── Dataset  (all values sourced live from the loaded df)
        line("  📂  DATASET", "section")
        line("-" * 68, "sep")
        kv("Dataset path:",   self.dataset_path.get())
        kv("Total rows:",     f"{df.shape[0]:,}")
        kv("Total columns:",  f"{df.shape[1]}")
        kv("Feature count:",  f"{len(feature_names)}")
        kv("Target column:",  self.target_col.get())
        # Live target statistics
        tgt_data = pd.to_numeric(df[self.target_col.get()], errors="coerce").dropna()
        kv("  Target mean:",  f"{tgt_data.mean():.4f}")
        kv("  Target std:",   f"{tgt_data.std():.4f}")
        kv("  Target min:",   f"{tgt_data.min():.4f}")
        kv("  Target max:",   f"{tgt_data.max():.4f}")
        kv("Task type:",      "Classification" if is_clf else "Regression")
        kv("Train samples:",  f"{train_size:,}")
        kv("Test samples:",   f"{test_size:,}")
        kv("Missing vals (original):", str(df.isnull().sum().sum()))
        line()

        # ── Model config  (read LIVE from trained model, not hardcoded)
        line("  ⚙️   MODEL CONFIGURATION", "section")
        line("-" * 68, "sep")
        model_type = type(model).__name__
        p = model.get_params()
        kv("Model:",               model_type)
        kv("n_estimators:",        str(p.get("n_estimators", "—")))
        kv("max_depth:",           str(p.get("max_depth", "—")))
        kv("learning_rate:",       str(p.get("learning_rate", "—")))
        kv("subsample:",           str(p.get("subsample", "—")))
        kv("colsample_bytree:",    str(p.get("colsample_bytree", "—")))
        kv("objective:",           str(p.get("objective", "—")))
        kv("eval_metric:",         str(p.get("eval_metric", "—")))
        kv("random_state:",        str(p.get("random_state", "—")))
        line()

        # ── Metrics
        line("  📊  EVALUATION METRICS", "section")
        line("-" * 68, "sep")
        if is_clf:
            acc = metrics["accuracy"]
            tag = "good" if acc > 0.75 else "bad"
            kv("Accuracy:", f"{acc:.4f}  ({acc*100:.1f}%)", tag)
            line()
            line("  Classification Report:", "key")
            line(metrics["classification_report"], "note")
        else:
            r2   = metrics["r2"]
            rmse = metrics["rmse"]
            tag  = "good" if r2 > 0.6 else "bad"
            kv("R² Score:",  f"{r2:.4f}",   tag)
            kv("MAE:",       f"{metrics['mae']:.4f}")
            kv("MSE:",       f"{metrics['mse']:.4f}")
            kv("RMSE:",      f"{rmse:.4f}")
        line()

        # ── Top 5 features
        line("  🏆  TOP 10 IMPORTANT FEATURES", "section")
        line("-" * 68, "sep")
        scores = model.get_booster().get_score(importance_type="weight")
        mapped = {feature_names[int(k.replace("f", ""))]: v
                  for k, v in scores.items()}
        top10  = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (feat, score) in enumerate(top10, 1):
            kv(f"  #{rank:>2}  {feat[:28]}", f"{int(score)}")
        line()

        # ── Spirometry parameter stats
        line("  🫁  KEY SPIROMETRY PARAMETER STATS", "section")
        line("-" * 68, "sep")
        spiro_cols = [c for c in SPIRO_KEY_COLS if c in df.columns]
        for col in spiro_cols:
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            short = col.replace("Baseline_", "").replace("_", " ")
            kv(f"  {short[:28]}", f"mean={data.mean():.3f}  std={data.std():.3f}  "
                                   f"min={data.min():.2f}  max={data.max():.2f}")
        line()

        # ── Interpretation
        line("  💡  INTERPRETATION", "section")
        line("-" * 68, "sep")
        if is_clf:
            acc = metrics["accuracy"]
            if acc > 0.85:
                line("  ✅  Excellent classification performance. The model reliably", "good")
                line("     discriminates between spirometry outcome classes.", "good")
            elif acc > 0.70:
                line("  ⚠️  Moderate accuracy. Consider class balancing, feature", "value")
                line("     engineering, or hyperparameter tuning.", "value")
            else:
                line("  ❌  Low accuracy. Review class imbalance and feature quality.", "bad")
        else:
            r2 = metrics["r2"]
            if r2 > 0.85:
                line("  ✅  Excellent regression fit (R² > 0.85). XGBoost captures", "good")
                line("     most variance in spirometry measurements.", "good")
            elif r2 > 0.60:
                line("  ⚠️  Good fit (R² > 0.60). Model has reasonable predictive", "value")
                line("     power for spirometry outcomes.", "value")
            else:
                line("  ❌  Weak fit. Target may require additional features or a", "bad")
                line("     different modelling strategy.", "bad")
        line()
        line("  Clinical relevance: FEV1, FVC, and their ratio are the primary", "note")
        line("  lung function indicators. Feature importance rankings reveal which", "note")
        line("  demographic and anthropometric factors most influence predictions.", "note")
        line()
        line("=" * 68, "sep")

        txt.config(state="disabled")

    # ════════════════════════════════════════════════════════════════════
    #  SAVE ALL PLOTS
    # ════════════════════════════════════════════════════════════════════

    def _save_plots(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        folder = filedialog.askdirectory(title="Select folder to save plots")
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        # Re-render and save each plot as PNG
        self._save_feature_importance(folder)
        self._save_pred_vs_actual(folder)
        self._save_residuals(folder)
        messagebox.showinfo("Saved", f"All plots saved to:\n{folder}")

    def _save_feature_importance(self, folder):
        scores = self.model.get_booster().get_score(importance_type="weight")
        mapped = {self.feature_names[int(k.replace("f", ""))]: v
                  for k, v in scores.items()}
        top20  = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:20]
        names  = [x[0] for x in top20][::-1]
        vals   = [x[1] for x in top20][::-1]
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(names))]
        ax.barh(names, vals, color=colors, edgecolor="none")
        ax.set_title("Feature Importance (Top 20)", color=TEXT_PRIMARY,
                     fontsize=12, fontweight="bold")
        ax.tick_params(colors=TEXT_SECONDARY)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_COLOR)
        ax.grid(True, axis="x", color=PLOT_GRID, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "feature_importance.png"),
                    dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)

    def _save_pred_vs_actual(self, folder):
        if self.is_clf:
            return
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.scatter(self.y_te, self.y_pred, alpha=0.35, s=8, color=ACCENT_BLUE)
        mn, mx = float(self.y_te.min()), float(self.y_te.max())
        ax.plot([mn, mx], [mn, mx], color=ACCENT_RED, linestyle="--", linewidth=1.5)
        ax.set_title("Predicted vs Actual", color=TEXT_PRIMARY, fontsize=11)
        ax.tick_params(colors=TEXT_SECONDARY)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_COLOR)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "prediction_vs_actual.png"),
                    dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)

    def _save_residuals(self, folder):
        if self.is_clf:
            return
        residuals = np.array(self.y_te) - np.array(self.y_pred)
        fig, ax   = plt.subplots(figsize=(8, 6), facecolor=PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.scatter(self.y_pred, residuals, alpha=0.35, s=8, color=ACCENT_PURPLE)
        ax.axhline(0, color=ACCENT_RED, linestyle="--", linewidth=1.5)
        ax.set_title("Residual Plot", color=TEXT_PRIMARY, fontsize=11)
        ax.tick_params(colors=TEXT_SECONDARY)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_COLOR)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "residual_plot.png"),
                    dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = SpiroDashboard()
    app.mainloop()
