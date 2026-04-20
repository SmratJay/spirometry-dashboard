import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("final_training_dataset.csv", low_memory=False)

print("Dataset Shape:", df.shape)
print(df.head())
print(df.columns)

# create FEV1/FVC ratio
df["FEV1_FVC"] = df["Baseline_FEV1_L"] / df["Baseline_FVC_L"]

# example additional feature
df["Lung_Index"] = df["Baseline_FEV1_L"] / df["Age"]

# 2. Select Features & Target
# ---------------------------

# Select useful features
features = [
    "Age",
    "Height2_x_Race",
    "Baseline_FEV1_L",
    "Baseline_FVC_L",
    "Lung_Index"
]

# Target label (clinical obstruction classification)
target = "Obstruction"

# select features
df = df[features + [target]]

# handle missing values
df = df.fillna(df.mean(numeric_only=True))

print("Rows after preprocessing:", len(df))

# ---------------------------
# 3. Features & Target Split
# ---------------------------
X = df[features]
y = df[target]

# ---------------------------
# 4. Train Test Split
# ---------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5. Train Random Forest
# ---------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# 6. Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 7. Evaluation
# ---------------------------
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
print(confusion_matrix(y_test, y_pred))
# ==============================
# Confusion Matrix Heatmap
# ==============================
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==============================
# ROC Curve
# ==============================
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ==============================
# Precision-Recall Curve
# ==============================
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# ==============================
# Sorted Feature Importance
# ==============================
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Sorted Feature Importance")
plt.show()