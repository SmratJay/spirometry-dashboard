# Complete Spirometer System Architecture

## Your Question Answered ✓

**Q: "How do we take real-time values from Arduino and convert to FEV1, FVC, and other respiratory data, then save to a file?"**

**A: Here's your complete 3-layer architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          LAYER 1: HARDWARE                             │
│                     (Arduino/ESP32 + Sensors)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    VL53L0X Sensor 1        VL53L0X Sensor 2        VL53L0X Sensor 3   │
│         (Tube 1)                (Tube 2)                (Tube 3)       │
│       Max 600cc              Max 900cc              Max 1200cc         │
│           │                       │                       │            │
│           └───────────────────────┴───────────────────────┘            │
│                                   │                                    │
│                     Reads ball position (mm)                           │
│                                   │                                    │
│                    Prints to Serial Port:                              │
│                   "S1:150  S2:240  S3:180"                             │
│                                   │                                    │
│              115200 baud, Every ~500ms                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │  spirometer_serial_reader.py              │
              │  (Python - Your Data Bridge)             │
              └──────────────────┬───────────────────────┘
                                 │
┌────────────────────────────────────────────────────────────────────────┐
│              LAYER 2: DATA ACQUISITION & CONVERSION                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1️⃣  CONNECT TO SERIAL PORT (COM5)                                    │
│      └─ Import pyserial library                                       │
│      └─ Opens communication with ESP32                                │
│                                                                        │
│  2️⃣  READ DATA FOR 6 SECONDS                                         │
│      └─ Every 100ms: Read "S1:XXX  S2:XXX  S3:XXX"                   │
│      └─ Store ~60 data points                                         │
│                                                                        │
│  3️⃣  CONVERT DISTANCE → VOLUME  ⭐ KEY STEP                           │
│      ┌─────────────────────────────────────┐                         │
│      │ INPUT:  Sensor distance (mm)        │                         │
│      │         S1=150mm, S2=240mm, S3=180mm│                         │
│      │                                     │                         │
│      │ FORMULA:                            │                         │
│      │ Volume = (Distance/Max_Distance)    │                         │
│      │          × Max_Volume               │                         │
│      │                                     │                         │
│      │ CALCULATION:                        │                         │
│      │ S1 = (150/200) × 600 = 450 mL     │ ← Tube 1                │
│      │ S2 = (240/200) × 900 = 1080 mL    │ ← Tube 2                │
│      │ S3 = (180/200) × 1200 = 1080 mL   │ ← Tube 3                │
│      │                                     │                         │
│      │ OUTPUT: Volume in mL for each tube │                         │
│      └─────────────────────────────────────┘                         │
│                                                                        │
│  4️⃣  SAVE RAW DATA (Time-series CSV)                                  │
│      ┌────────────────────────────────────┐                          │
│      │ spirometry_PATIENT_001_TIMESTAMP.csv│                         │
│      ├────────────────────────────────────┤                          │
│      │Time   │Volume1│Volume2│Volume3     │                          │
│      ├────────────────────────────────────┤                          │
│      │0.00   │45.3   │67.2   │89.1        │                          │
│      │0.10   │102.4  │156.8  │198.5       │                          │
│      │0.20   │189.7  │285.3  │412.6       │                          │
│      │...    │...    │...    │...         │                          │
│      │6.00   │612.4  │834.6  │1120.5      │                          │
│      └────────────────────────────────────┘                          │
│                             │                                        │
│                             ▼                                        │
│                                                                      │
│  5️⃣  CALCULATE RESPIRATORY METRICS  ⭐ ANALYSIS                      │
│                                                                      │
│      From Volume-Time Curve, extract:                               │
│      ┌─────────────────────────────────────┐                        │
│      │ FVC = Maximum Volume Reached        │                        │
│      │       (patient's total lung capacity)│                       │
│      │                                      │                        │
│      │ FEV1 = Volume at exactly 1 second   │                        │
│      │        (forced expiration in 1 sec) │                        │
│      │                                      │                        │
│      │ FEV1/FVC = Diagnostic Ratio          │                        │
│      │  < 70% → likely COPD                │                        │
│      │  70-80% → mild obstruction          │                        │
│      │  > 80% → normal lung function       │                        │
│      │                                      │                        │
│      │ PEF = Peak Expiratory Flow           │                        │
│      │       (max rate of breathing out)   │                        │
│      └─────────────────────────────────────┘                        │
│                                                                      │
│  6️⃣  SAVE RESULTS (Metrics CSV)                                      │
│      ┌────────────────────────────────────────┐                     │
│      │ spirometry_results_PATIENT_001_*.csv    │                     │
│      ├────────────────────────────────────────┤                     │
│      │Sensor │FVC    │FEV1   │FEV1/FVC│PEF   │                     │
│      ├────────────────────────────────────────┤                     │
│      │Tube 1 │612.4  │489.2  │79.9%   │245.6 │                     │
│      │Tube 2 │834.6  │689.3  │82.6%   │310.2 │                     │
│      │Tube 3 │1120.5 │798.4  │71.3%   │280.5 │                     │
│      └────────────────────────────────────────┘                     │
│                                                                      │
└────────────────────────────────────┬─────────────────────────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │  example_use_test_data.py                 │
              │  (Python - Data Integration)             │
              └──────────────────┬───────────────────────┘
                                 │
┌────────────────────────────────────────────────────────────────────────┐
│         LAYER 3: MACHINE LEARNING & CLINICAL ANALYSIS                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  📊 LOAD YOUR CSV FILES INTO PANDAS                                  │
│     raw_data = pd.read_csv("spirometry_PATIENT_001_*.csv")           │
│     metrics = pd.read_csv("spirometry_results_PATIENT_001_*.csv")     │
│                                                                        │
│  🔍 COMPARE TO NHANES REFERENCE DATABASE                             │
│     - Load NHANES 2007-2012 data (you already have this!)            │
│     - Find patient's percentile rank                                  │
│     - Example: "FEV1 is at 65th percentile for age/gender"          │
│                                                                        │
│  🤖 FEED INTO ML MODELS (combined_dashboard_v4.py)                   │
│     ┌──────────────────────────────────────┐                         │
│     │ Extract Features from Test:          │                         │
│     │ • Max volume per tube                │                         │
│     │ • Mean volume                        │                         │
│     │ • Variability (std dev)              │                         │
│     │ • Max flow rate (PEF)                │                         │
│     │ • FEV1/FVC ratio                     │                         │
│     └───────────────┬──────────────────────┘                         │
│                     │                                                │
│                     ▼                                                │
│     ┌──────────────────────────────────────┐                         │
│     │ Feed to ML Models:                   │                         │
│     │ • XGBoost                            │                         │
│     │ • Random Forest                      │                         │
│     │ • SVM (RBF)                          │                         │
│     │ • MLP Neural Network                 │                         │
│     └───────────────┬──────────────────────┘                         │
│                     │                                                │
│                     ▼                                                │
│     ┌──────────────────────────────────────────┐                    │
│     │ PREDICTIONS & DIAGNOSIS:                 │                    │
│     │ • FEV1/FVC classification               │                    │
│     │ • COPD risk assessment                  │                    │
│     │ • Asthma severity prediction            │                    │
│     │ • Track breathing improvement over time │                    │
│     └──────────────────────────────────────────┘                    │
│                                                                      │
│  📈 VISUALIZE RESULTS                                               │
│     • Plot volume vs. time (breathing curve)                        │
│     • Feature importance (which tube matters most?)                 │
│     • Model predictions vs. NHANES                                  │
│                                                                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Execution

### ✅ Step 0: Have Arduino Code Running
```cpp
// Your Arduino already does this:
// 1. Reads 3 sensors
// 2. Converts distance to readable output
// 3. Prints: "S1:150  S2:240  S3:180"
```
**Status**: ✓ Done

---

### ✅ Step 1: Capture Real Test Data
```powershell
# Terminal 1: Run the serial reader
python spirometer_serial_reader.py
```

**What happens:**
- Connects to COM5 (your ESP32)
- Waits for patient to blow
- Reads for 6 seconds
- Saves: `spirometry_PATIENT_001_20260420_143022.csv`
- Saves: `spirometry_results_PATIENT_001_20260420_143022.csv`

**Output:**
```
T=0.1s | S1=45.3mL | S2=67.2mL | S3=89.1mL
T=0.2s | S1=102.4mL | S2=156.8mL | S3=198.5mL
T=0.3s | S1=189.7mL | S2=285.3mL | S3=412.6mL
...
✓ Test complete!
✓ Saved to spirometry_PATIENT_001_20260420_143022.csv
✓ Results saved to spirometry_results_PATIENT_001_20260420_143022.csv
```

---

### ✅ Step 2: Review Your Data Files

**Raw Data (Time-series):**
```csv
Time (s),Sensor1_Volume (mL),Sensor2_Volume (mL),Sensor3_Volume (mL)
0.00,45.3,67.2,89.1
0.10,102.4,156.8,198.5
0.20,189.7,285.3,412.6
...
6.00,612.4,834.6,1120.5
```
➜ Use for: Visualizing breath pattern, detecting anomalies

**Results (Metrics):**
```csv
Sensor,FVC (mL),FEV1 (mL),FEV1/FVC %,PEF (mL/s),Duration (s)
Sensor 1,612.4,489.2,79.9,245.6,6.00
Sensor 2,834.6,689.3,82.6,310.2,6.00
Sensor 3,1120.5,798.4,71.3,280.5,6.00
```
➜ Use for: Clinical interpretation, ML model input

---

### ✅ Step 3: Integrate with Your Dashboard

```powershell
# Terminal 2: Load test data and analyze
python example_use_test_data.py
```

## Complete Data Conversion Example

Here's exactly what happens for **one measurement**:

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Arduino reads sensor at time T=1.5 seconds         │
├─────────────────────────────────────────────────────────────┤
│ VL53L0X Sensor 1 measures: 123 mm (ball height)            │
│ VL53L0X Sensor 2 measures: 178 mm (ball height)            │
│ VL53L0X Sensor 3 measures: 142 mm (ball height)            │
│                                                              │
│ Arduino prints:                                              │
│ "S1:123  S2:178  S3:142"                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Python serial_reader.py receives the string        │
├─────────────────────────────────────────────────────────────┤
│ Parses: S1=123, S2=178, S3=142                              │
│ Timestamp: T=1.5 seconds                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Convert distance (mm) → volume (mL)                │
├─────────────────────────────────────────────────────────────┤
│ For Sensor 1 (600cc max tube):                              │
│   Volume1 = (123/200) × 600 = 369 mL                        │
│                                                              │
│ For Sensor 2 (900cc max tube):                              │
│   Volume2 = (178/200) × 900 = 801 mL                        │
│                                                              │
│ For Sensor 3 (1200cc max tube):                             │
│   Volume3 = (142/200) × 1200 = 852 mL                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Store in CSV (for all 6 seconds of data)           │
├─────────────────────────────────────────────────────────────┤
│ CSV Row:                                                     │
│ 1.50, 369, 801, 852                                         │
│                                                              │
│ (Along with 59 other rows from T=0.0s to T=6.0s)           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Analyze entire test to find FEV1, FVC, PEF        │
├─────────────────────────────────────────────────────────────┤
│ From all 60 data points:                                    │
│                                                              │
│ FVC = Max(Volume1) across all 6 seconds = 612.4 mL         │
│ FEV1 = Volume1 at T=1.0 second = 489.2 mL                  │
│ FEV1/FVC = 489.2 / 612.4 = 79.9%                           │
│ PEF = Max(flow rate) = 245.6 mL/s                          │
└─────────────────────────────────────────────────────────────┘
```

---

## File Organization

Your workspace should now have:

```
d:\New folder (2)\
├── combined_dashboard_v4.py              ← Main dashboard (uses NHANES)
├── spirometer_serial_reader.py           ← NEW: Capture test data
├── example_use_test_data.py              ← NEW: Use test data
├── SPIROMETRY_DATA_PIPELINE.md           ← NEW: This guide
│
├── NHANES_2007_2012_*.csv                ← Reference data
├── final_training_dataset.csv            ← Training data
│
├── spirometry_PATIENT_001_*.csv          ← Test 1 (raw data)
├── spirometry_results_PATIENT_001_*.csv  ← Test 1 (metrics)
│
├── spirometry_PATIENT_002_*.csv          ← Test 2 (raw data)
├── spirometry_results_PATIENT_002_*.csv  ← Test 2 (metrics)
│
└── all_tests_summary.csv                 ← (Generated by batch processing)
```

---

## Summary: Your Data's Journey

```
Arduino Sensor → Distance (mm)
                 ↓
            Python Script
                 ↓
        Volume (mL) per tube
                 ↓
        Time-series CSV saved
                 ↓
        Analyze 6-second curve
                 ↓
        Calculate FEV1, FVC, PEF
                 ↓
        Metrics CSV saved
                 ↓
        Compare to NHANES
                 ↓
        Feed to ML Models
                 ↓
        Clinical Diagnosis & Report
```

You now have the **complete data pipeline** from hardware to clinical analysis! 🎉

---

**Next: Run it!**
```powershell
# Terminal 1
python spirometer_serial_reader.py

# Terminal 2 (after test completes)
python example_use_test_data.py
```

Questions? See `SPIROMETRY_DATA_PIPELINE.md` for troubleshooting.
