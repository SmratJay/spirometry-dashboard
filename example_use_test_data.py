"""
Example: Integrate Spirometer Test Data with Dashboard
Shows how to load real test data and use it with your ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# ============================================================================
# LOAD TEST DATA
# ============================================================================

def load_spirometer_test(raw_data_csv, results_csv=None):
    """
    Load real spirometer test data
    
    Parameters:
        raw_data_csv: Path to spirometry_PATIENT_XXX_TIMESTAMP.csv
        results_csv: Path to spirometry_results_PATIENT_XXX_TIMESTAMP.csv
    
    Returns:
        dict with raw data and metrics
    """
    
    # Load raw time-series data
    raw_df = pd.read_csv(raw_data_csv)
    
    # Load metrics (optional)
    metrics = None
    if results_csv and os.path.exists(results_csv):
        metrics = pd.read_csv(results_csv)
    
    return {
        'raw_data': raw_df,
        'metrics': metrics,
        'filename': Path(raw_data_csv).name
    }


# ============================================================================
# EXTRACT TEST FEATURES FOR ML MODEL
# ============================================================================

def create_ml_features_from_test(raw_data):
    """
    Convert spirometer test data to features for ML model prediction
    
    This shows how to prepare real patient data for your dashboard's
    ML models (XGBoost, Random Forest, SVM, MLP)
    """
    
    features = {}
    
    # For each sensor (3 tubes)
    for sensor_col in ['Sensor1_Volume (mL)', 'Sensor2_Volume (mL)', 'Sensor3_Volume (mL)']:
        if sensor_col not in raw_data.columns:
            continue
        
        volumes = raw_data[sensor_col].values
        
        # Extract features that ML models might need:
        features[f'{sensor_col}_max'] = np.max(volumes)              # FVC
        features[f'{sensor_col}_min'] = np.min(volumes)              # Residual volume
        features[f'{sensor_col}_mean'] = np.mean(volumes)            # Average
        features[f'{sensor_col}_std'] = np.std(volumes)              # Variability
        features[f'{sensor_col}_range'] = np.max(volumes) - np.min(volumes)  # Total capacity
        
        # Flow features (slope)
        if len(volumes) > 1:
            flow = np.diff(volumes)
            features[f'{sensor_col}_max_flow'] = np.max(np.abs(flow))  # PEF
            features[f'{sensor_col}_mean_flow'] = np.mean(np.abs(flow))
    
    return features


# ============================================================================
# WORKFLOW EXAMPLE
# ============================================================================

def example_workflow():
    """
    Complete example: Load test data and prepare for dashboard
    """
    
    print("="*70)
    print("SPIROMETER TEST DATA INTEGRATION EXAMPLE")
    print("="*70)
    
    # STEP 1: Find your test data
    # ─────────────────────────────────────────────────────────────────────
    current_dir = Path(".")
    test_files = list(current_dir.glob("spirometry_*.csv"))
    
    if not test_files:
        print("\n✗ No test data found in current directory")
        print("   Run spirometer_serial_reader.py first to capture a test\n")
        return None
    
    print(f"\n✓ Found {len(test_files)} test file(s):\n")
    for i, f in enumerate(test_files, 1):
        print(f"  {i}. {f.name}")
    
    # STEP 2: Load the most recent test
    # ─────────────────────────────────────────────────────────────────────
    latest_raw = sorted(list(current_dir.glob("spirometry_PATIENT_*.csv")))[-1]
    latest_results = sorted(list(current_dir.glob("spirometry_results_PATIENT_*.csv")))[-1] \
        if list(current_dir.glob("spirometry_results_PATIENT_*.csv")) else None
    
    print(f"\n▶ Loading most recent test: {latest_raw.name}")
    
    test_data = load_spirometer_test(latest_raw, latest_results)
    
    # STEP 3: Inspect raw data
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RAW DATA SUMMARY")
    print("="*70)
    print(test_data['raw_data'].head(10))
    print(f"\nShape: {test_data['raw_data'].shape}")
    print(f"Time range: {test_data['raw_data']['Time (s)'].min():.2f}s to {test_data['raw_data']['Time (s)'].max():.2f}s")
    
    # STEP 4: View metrics
    # ─────────────────────────────────────────────────────────────────────
    if test_data['metrics'] is not None:
        print("\n" + "="*70)
        print("CALCULATED METRICS")
        print("="*70)
        print(test_data['metrics'].to_string(index=False))
    
    # STEP 5: Extract ML features
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ML FEATURES EXTRACTED FROM TEST")
    print("="*70)
    
    features = create_ml_features_from_test(test_data['raw_data'])
    
    # Display as pandas Series for clarity
    features_series = pd.Series(features)
    print(features_series.to_string())
    
    # STEP 6: Prepare for your dashboard
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("HOW TO USE WITH YOUR DASHBOARD")
    print("="*70)
    print("""
1. TRAINING DATA:
   → Add this test's raw CSV to your NHANES training dataset
   → Includes: Time, Sensor1_Volume, Sensor2_Volume, Sensor3_Volume
   
2. FEATURES FOR ML:
   → Use the extracted features (max, mean, std, flow) as model inputs
   → Helps XGBoost, Random Forest predict respiratory health
   
3. VISUALIZATION:
   → Plot raw_data columns over Time to show breathing curve
   → Compare multiple tests to detect improvements/decline
   
4. REFERENCE COMPARISON:
   → Load NHANES dataset from your dashboard
   → Compare patient's FEV1/FVC ratio against reference population
   → Calculate percentile rank
    """)
    
    # STEP 7: Example prediction (if you have a trained model)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("NEXT: INTEGRATE WITH YOUR XGBOOST MODEL")
    print("="*70)
    
    print("""
In combined_dashboard_v4.py, you could add:

    # Load real test data
    test_csv = "spirometry_PATIENT_001_20260420_143022.csv"
    test_data = pd.read_csv(test_csv)
    
    # Extract features
    X_test = pd.DataFrame([create_ml_features_from_test(test_data)])
    
    # Predict with your trained XGBoost model
    if hasattr(self, 'model_xgb'):
        prediction = self.model_xgb.predict(X_test)[0]
        print(f"Predicted FEV1: {prediction:.2f} mL")
        
        # Compare to NHANES reference
        percentile = (self.df[self.df['FEV1'] < prediction].shape[0] / 
                     len(self.df) * 100)
        result_text = f"Patient is at {percentile:.1f}th percentile"
    """)
    
    return test_data, features


# ============================================================================
# BATCH PROCESSING: Multiple Tests
# ============================================================================

def process_all_tests(output_csv="all_tests_summary.csv"):
    """
    Process all spirometer test files in directory
    Returns summary of all tests
    """
    
    current_dir = Path(".")
    test_files = sorted(list(current_dir.glob("spirometry_PATIENT_*.csv")))
    
    all_tests = []
    
    for test_file in test_files:
        results_file = test_file.parent / test_file.name.replace(
            "spirometry_", "spirometry_results_")
        
        test = load_spirometer_test(test_file, results_file)
        
        summary = {
            'test_file': test_file.name,
            'timestamp': test_file.name.split('_')[-1],
        }
        
        # Add metrics if available
        if test['metrics'] is not None:
            for _, row in test['metrics'].iterrows():
                summary[f"Sensor{row['Sensor'].split()[-1]}_FVC"] = row['FVC (mL)']
                summary[f"Sensor{row['Sensor'].split()[-1]}_FEV1"] = row['FEV1 (mL)']
                summary[f"Sensor{row['Sensor'].split()[-1]}_Ratio"] = row['FEV1/FVC %']
        
        all_tests.append(summary)
    
    # Save summary
    summary_df = pd.DataFrame(all_tests)
    summary_df.to_csv(output_csv, index=False)
    
    print(f"✓ Saved summary of {len(all_tests)} tests to {output_csv}")
    return summary_df


# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    test_data, features = example_workflow()
    
    print("\n" + "="*70)
    print("✓ EXAMPLE COMPLETE - Your test data is ready for analysis!")
    print("="*70)
