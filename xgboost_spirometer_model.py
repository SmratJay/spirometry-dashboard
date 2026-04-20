import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import warnings
from sklearn.exceptions import DataConversionWarning
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Global Configuration ---
DATASET_PATH = r"D:\New folder (2)\NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"
TARGET_COLUMN = "FEV1" # IMPORTANT: Change this to your actual target column name
RANDOM_STATE = 42

def load_data(file_path):
    """
    Loads the dataset from a CSV file, handles missing values, and displays basic information.
    """
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        exit()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 Rows:\n", df.head())
    print("\nColumn Names:\n", df.columns.tolist())

    # Handle missing values: For simplicity, fill numerical NaNs with median and categorical with mode.
    # A more sophisticated approach might involve imputation based on domain knowledge or more advanced techniques.
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                print(f"Filled missing numerical values in column '{col}' with median.")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"Filled missing categorical values in column '{col}' with mode.")

    print("\nMissing values after handling:\n", df.isnull().sum())
    return df

def preprocess_data(df, target_column):
    """
    Preprocesses the data by separating features and target, detecting column types,
    encoding categorical features, and scaling numerical features.
    """
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        exit()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Automatically detect numerical vs categorical columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Drop any potential ID columns - assuming columns named 'ID' or 'id'
    id_cols = [col for col in X.columns if 'id' in col.lower()]
    if id_cols:
        print(f"Dropping potential ID columns: {id_cols}")
        X = X.drop(columns=id_cols)
        numerical_cols = [col for col in numerical_cols if col not in id_cols]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]

    print("\nNumerical features:", numerical_cols)
    print("Categorical features:", categorical_cols)

    # Determine if it's a classification or regression problem based on the target column
    is_classification = False
    if pd.api.types.is_numeric_dtype(y):
        # If numerical, check number of unique values. If few and integer-like, assume classification.
        if y.nunique() <= 20 and y.dtype in ['int64', 'int32', 'int16']: # Arbitrary threshold for classification
            print(f"Target column '{target_column}' seems like a classification target (numerical with few unique integer values).")
            is_classification = True
        else:
            print(f"Target column '{target_column}' is a regression target (numerical with many unique values).")
            is_classification = False
    else:
        print(f"Target column '{target_column}' is a classification target (non-numerical).")
        is_classification = True

    # Encode target if classification
    if is_classification and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Target column '{target_column}' encoded using LabelEncoder.")

    # Encode categorical features
    if categorical_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = ohe.fit_transform(X[categorical_cols])
        encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
        print("Categorical features encoded using OneHotEncoder.")

    # Scale numerical features
    if numerical_cols:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        print("Numerical features scaled using StandardScaler.")

    print("\nFeatures (X) after preprocessing shape:", X.shape)
    print("Target (y) after preprocessing shape:", y.shape)
    print("\nFeatures after preprocessing:\n", X.head())

    return X, y, is_classification

def train_model(X_train, y_train, is_classification):
    """
    Trains an XGBoost model (Classifier or Regressor) with predefined parameters.
    """
    print("\nTraining XGBoost model...")
    if is_classification:
        model = xgb.XGBClassifier(
            objective='binary:logistic' if y_train.nunique() == 2 else 'multi:softmax',
            num_class=y_train.nunique() if y_train.nunique() > 2 else None,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False, # Suppress warning
            eval_metric='logloss', # Common metric for classification
            random_state=RANDOM_STATE
        )
        print("Using XGBClassifier.")
    else:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE
        )
        print("Using XGBRegressor.")

    model.fit(X_train, y_train)
    print("Model training complete.")
    print("\nModel Parameters:\n", model.get_params())
    return model

def evaluate_model(model, X_test, y_test, is_classification):
    """
    Evaluates the trained model using appropriate metrics.
    """
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)

    if is_classification:
        # Convert probabilities to class labels for binary classification if needed
        if hasattr(model, 'predict_proba') and y_test.nunique() == 2:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int) # Example threshold
        else: # For multi-class or already predicted labels
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0) # Added zero_division to avoid warnings

        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)
        return y_pred, {'accuracy': accuracy, 'confusion_matrix': cm, 'classification_report': report}
    else:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        return y_pred, {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2_score': r2}

def plot_results(model, X_test, y_test, y_pred, is_classification, feature_names, metrics):
    """
    Generates and saves feature importance, prediction vs actual, or confusion matrix plots.
    """
    print("\nGenerating plots...")
    
    # Ensure directories exist for saving plots
    os.makedirs("plots", exist_ok=True)

    # 1. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    xgb.plot_importance(model, ax=ax, importance_type='weight', max_num_features=10) # Using weight as default importance type
    plt.title("XGBoost Feature Importance (Top 10)")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close(fig)
    print("Feature importance plot saved as 'plots/feature_importance.png'")

    if is_classification:
        # 2. Confusion Matrix Visualization
        cm = metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='red')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig("plots/confusion_matrix.png")
        plt.close(fig)
        print("Confusion matrix plot saved as 'plots/confusion_matrix.png'")
    else:
        # 2. Prediction vs Actual Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Prediction vs Actual Values")
        plt.tight_layout()
        plt.savefig("plots/prediction_vs_actual.png")
        plt.close(fig)
        print("Prediction vs Actual scatter plot saved as 'plots/prediction_vs_actual.png'")

        # 3. Residual Plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        plt.tight_layout()
        plt.savefig("plots/residual_plot.png")
        plt.close(fig)
        print("Residual plot saved as 'plots/residual_plot.png'")
    print("All plots generated and saved in 'plots/' directory.")

def main():
    """
    Main function to run the entire ML pipeline.
    """
    # Set reproducibility seeds
    np.random.seed(RANDOM_STATE)
    
    print("--- Starting XGBoost ML Pipeline ---")

    # 1. Load Data
    df = load_data(DATASET_PATH)

    # 2. Preprocessing
    X, y, is_classification = preprocess_data(df.copy(), TARGET_COLUMN) # Pass a copy to avoid modifying original df

    # Prepare feature names for XGBoost (important for plotting feature importance)
    feature_names = X.columns.tolist()
    X.columns = [f"f{i}" for i in range(X.shape[1])] # XGBoost prefers simple feature names

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if is_classification else None)
    print(f"\nTrain/Test Split: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

    # 4. Model Training
    model = train_model(X_train, y_train, is_classification)

    # 5. Evaluation
    y_pred, metrics = evaluate_model(model, X_test, y_test, is_classification)

    # 6. Feature Importance and Visualization Plots
    plot_results(model, X_test, y_test, y_pred, is_classification, feature_names, metrics)

    # 7. Conclusion Output
    print("\n" + "="*30)
    print("MODEL TRAINING SUMMARY")
    print("="*30)
    print(f"- Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"- Number of features: {X.shape[1]}")
    model_type = "XGBoost Classifier" if is_classification else "XGBoost Regressor"
    print(f"- Model used: {model_type}")

    print("- Key metrics:")
    if is_classification:
        print(f"    - Accuracy: {metrics['accuracy']:.4f}")
        print("    - Classification Report (excerpt):\n", "\n".join(metrics['classification_report'].split('\n')[0:5])) # Print top few lines
    else:
        print(f"    - MAE: {metrics['mae']:.4f}")
        print(f"    - MSE: {metrics['mse']:.4f}")
        print(f"    - RMSE: {metrics['rmse']:.4f}")
        print(f"    - R2 Score: {metrics['r2_score']:.4f}")

    # Get top 5 important features (using the original feature names)
    importance_scores = model.get_booster().get_score(importance_type='weight')
    # Map 'fI' back to original feature names
    original_feature_importance = {feature_names[int(k.replace('f', ''))]: v for k, v in importance_scores.items()} if feature_names else {}
    sorted_features = sorted(original_feature_importance.items(), key=lambda item: item[1], reverse=True)
    top_5_features = sorted_features[:5]
    print("- Top 5 important features:")
    for feature, score in top_5_features:
        print(f"    - {feature}: {score:.2f}")

    print("\n- Brief interpretation of model performance:")
    if is_classification:
        if metrics['accuracy'] > 0.75: # Example threshold
            print("    The classification model shows good predictive accuracy, suggesting it can distinguish between classes effectively.")
            print("    Further analysis of the confusion matrix and precision/recall scores can provide deeper insights into specific class performance.")
        else:
            print("    The classification model's accuracy is moderate. It might benefit from hyperparameter tuning, more data, or feature engineering.")
            print("    Reviewing the confusion matrix can highlight specific areas of misclassification.")
    else:
        if metrics['r2_score'] > 0.6: # Example threshold
            print("    The regression model explains a good portion of the variance in the target variable, indicating reasonable predictive power.")
            print("    The MAE and RMSE values suggest the average prediction error, which appears acceptable for the given scale of the target.")
        else:
            print("    The regression model has limited predictive power, as indicated by the R2 score. Consider improving features, trying different models, or tuning hyperparameters.")
            print("    The residual plot can help identify patterns in errors, indicating potential model biases or non-linear relationships.")
    print("\n--- ML Pipeline Finished ---")

if __name__ == "__main__":
    main()
