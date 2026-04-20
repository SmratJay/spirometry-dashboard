# Quick Reference: Spirometer Data Pipeline

## 📋 What Happens at Each Stage

| Stage | Input | Process | Output | File |
|-------|-------|---------|--------|------|
| **Hardware** | Air blown | VL53L0X measures distance | Distance (mm) | Serial port |
| **Capture** | Serial data | Read 6 seconds | Times + distances | Console |
| **Convert** | Distance (mm) | Formula: V = (D/Max) × Vol | Volume (mL) per tube | CSV (raw) |
| **Analyze** | Volume curve | Find peaks & slopes | FEV1, FVC, PEF | CSV (results) |
| **Dashboard** | CSV files | Load + visualize | ML predictions | Dashboard UI |

---

## 🚀 Quick Start (3 Commands)

```powershell
# 1. Install serial library
pip install pyserial

# 2. Run test (person blows)
python spirometer_serial_reader.py

# 3. Analyze data
python example_use_test_data.py
```

---

## 🔄 The Calibration Formula

Your hardware → software conversion:

```
Volume (mL) = (Distance_mm / Max_Distance_mm) × Max_Volume_mL

Example for Tube 1:
- Sensor reads: 150 mm
- Max distance: 200 mm  
- Max volume: 600 mL
- Result: (150/200) × 600 = 450 mL
```

**Calibration Values Used:**
- Sensor 1: 600 mL maximum
- Sensor 2: 900 mL maximum  
- Sensor 3: 1200 mL maximum
- Max distance: 200 mm

⚠️ **Adjust these in `spirometer_serial_reader.py` lines 29-33 if your tubes are different!**

---

## 📊 Respiratory Metrics Explained

| Metric | What It Means | Normal Value | Clinical Use |
|--------|--------------|--------------|--------------|
| **FVC** | Total lung capacity when breathing out hard | 3-5 L (adults) | Overall lung size |
| **FEV1** | How much air exhaled in 1st second | 2.5-4 L (adults) | Breathing power |
| **FEV1/FVC** | Ratio (%) | > 70% | Detects obstruction |
| **PEF** | Peak breathing-out speed | 300-600 mL/s | Asthma indicator |

**Example Reading:**
- FVC = 612 mL = Patient can exhale 612 mL total
- FEV1 = 489 mL = In 1 second, 489 mL comes out
- Ratio = 489/612 = 79.9% = **Normal** (> 70% is good)

---

## 📁 File Reference

### Input Files (You Already Have)
```
NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv  - Reference data
combined_dashboard_v4.py                                  - ML dashboard
```

### New Files Created
```
spirometer_serial_reader.py          - Reads hardware, converts, saves
example_use_test_data.py             - Loads CSV, analyzes
SPIROMETRY_DATA_PIPELINE.md          - Detailed guide
COMPLETE_SPIROMETER_ARCHITECTURE.md  - Full architecture diagram
```

### Generated Files (Per Test)
```
spirometry_PATIENT_001_20260420_143022.csv          - Raw time-series
spirometry_results_PATIENT_001_20260420_143022.csv  - Calculated metrics
all_tests_summary.csv                               - Batch results
```

---

## 🔧 Customization Quick Tips

### Change Test Duration
**File:** `spirometer_serial_reader.py`, line 24
```python
TEST_DURATION = 6  # Change to 10, 15, etc.
```

### Change Serial Port
**File:** `spirometer_serial_reader.py`, line 23
```python
SERIAL_PORT = "COM3"  # Change to your port
```

### Adjust Calibration
**File:** `spirometer_serial_reader.py`, lines 29-35
```python
TUBE_MAX_VOLUMES = {
    'sensor1': 600,    # Change based on your tubes
    'sensor2': 900,
    'sensor3': 1200,
}
SENSOR_MAX_DISTANCE_MM = 200  # Change based on your sensor range
```

### Change Subject ID in Output
```python
result = run_test(subject_id="JOHN_DOE_001")
```

---

## ✅ Troubleshooting Checklist

| Problem | Check |
|---------|-------|
| "Failed to connect to COM3" | Device Manager → Which COM port is ESP32 on? |
| "No data received" | Arduino must print "S1:XXX  S2:XXX  S3:XXX" format |
| "File not found error" | Ensure CSV file is in same directory as Python script |
| Volumes too low/high | Adjust `TUBE_MAX_VOLUMES` and `SENSOR_MAX_DISTANCE_MM` |
| FEV1 = 0 | Patient didn't blow hard enough—re-run test |

---

## 📈 Complete Data Flow (1 Test)

```python
# Step 1: Hardware captures
D1 = 150mm, D2 = 240mm, D3 = 180mm  (at time T=0.1s)

# Step 2: Convert to volume
V1 = (150/200)*600 = 450mL
V2 = (240/200)*900 = 1080mL
V3 = (180/200)*1200 = 1080mL

# Step 3: Save row
CSV Row: 0.1, 450.0, 1080.0, 1080.0

# Step 4: Repeat 59 more times (0.1s to 6.0s)

# Step 5: Analyze all 60 rows
FVC = MAX(all volumes) = 612.4 mL
FEV1 = Volume at T=1.0s = 489.2 mL
FEV1/FVC = 489.2/612.4 = 79.9%

# Step 6: Save results
CSV: Sensor1, 612.4, 489.2, 79.9, 245.6, 6.0
```

---

## 🎯 What to Do Next

1. **✓ Install dependencies:**
   ```powershell
   pip install pyserial scipy
   ```

2. **✓ Verify Arduino is printing correctly** 
   - Open Arduino Serial Monitor
   - Confirm output: `"S1:XXX  S2:XXX  S3:XXX"`

3. **✓ Run your first test:**
   ```powershell
   python spirometer_serial_reader.py
   ```

4. **✓ Analyze the results:**
   ```powershell
   python example_use_test_data.py
   ```

5. **✓ Integrate with dashboard:**
   - Load CSV files into `combined_dashboard_v4.py`
   - Compare metrics to NHANES data
   - Train ML models on your test data

---

## 📞 Key Code References

### In `spirometer_serial_reader.py`:

```python
# Connection
reader = SpirometryReader(port="COM3", baud=115200)
reader.connect()

# Capture test
test_data = reader.read_test(duration=6)

# Save
reader.save_to_csv(test_data, subject_id="PATIENT_001")

# Analyze
metrics = calculate_respiratory_metrics(timestamps, volumes, sensor_id=1)
# Returns: fvc, fev1, fev1_fvc_ratio, pef
```

### In `example_use_test_data.py`:

```python
# Load test
test = load_spirometer_test("spirometry_PATIENT_001_*.csv")

# Get raw data
volumes = test['raw_data']['Sensor1_Volume (mL)'].values

# Extract features
features = create_ml_features_from_test(test['raw_data'])
```

---

## 🧬 Feature Engineering for ML

The serial reader extracts these features from each test:

```python
For each sensor (3 tubes):
- sensor_X_max        → Peak volume (FVC)
- sensor_X_min        → Residual volume
- sensor_X_mean       → Average lung volume
- sensor_X_std        → Variability
- sensor_X_range      → Total capacity used
- sensor_X_max_flow   → Peak expiratory flow (PEF)
- sensor_X_mean_flow  → Average breathing rate
```

Use these features in your XGBoost/Random Forest/SVM models!

---

## 🔄 Test Batch Processing

Process multiple tests at once:

```python
from example_use_test_data import process_all_tests

# Creates all_tests_summary.csv with metrics from all tests
summary = process_all_tests()
print(summary)
```

---

## Important Notes

⚠️ **Calibration is Critical**
- If your tubes have different volumes, adjust `TUBE_MAX_VOLUMES`
- If your sensor range is different, adjust `SENSOR_MAX_DISTANCE_MM`
- Test with a known volume (e.g., calibrated syringe) first

✅ **Sampling Rate**
- Default: 10 Hz (read every 100ms)
- Change in line 21: `SAMPLING_RATE = 10`
- Higher = more data points, larger files

✅ **Test Duration**
- Default: 6 seconds (standard for FEV1)
- Can extend to 15 seconds for complete FVC
- Change in line 24: `TEST_DURATION = 6`

✅ **Serial Format**
- Arduino MUST print: `"S1:123  S2:234  S3:345"`
- Exactly this format, every ~500ms
- Your Arduino code already does this ✓

---

## Performance Notes

| Metric | Value | Notes |
|--------|-------|-------|
| Data points per test | ~60 | At 10 Hz sampling, 6 seconds = 60 rows |
| File size per test | <1 KB | Very small CSV files |
| Processing time | <100ms | Minimal computation |
| Storage for 1000 tests | ~1 MB | Virtually no disk space needed |

---

## Questions? Read These Docs

1. **"How does calibration work?"** 
   → See `SPIROMETRY_DATA_PIPELINE.md` section "Calibration Formula"

2. **"What are the respiratory metrics?"**
   → See `COMPLETE_SPIROMETER_ARCHITECTURE.md` section "Respiratory Metrics Explained"

3. **"How do I test this without a patient?"**
   → Simulate with Arduino: send decreasing distances to simulate breathing

4. **"How do I integrate with my dashboard?"**
   → See `example_use_test_data.py` for loading and feature extraction

---

## Final Summary

```
Your Hardware               Your Software
───────────────────         ─────────────────────
Arduino/ESP32               spirometer_serial_reader.py
  ↓                             ↓
VL53L0X × 3                 Real-time conversion
  ↓                             ↓
mm distance                 mL volume
  ↓                             ↓
Serial output               CSV files (raw)
                                ↓
                            Analyze metrics
                                ↓
                            CSV files (results)
                                ↓
                            Dashboard + ML
```

You now have the **complete real-time spirometry data pipeline**! 🎉
