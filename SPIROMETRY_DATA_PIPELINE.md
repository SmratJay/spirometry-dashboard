# Real-Time Spirometer Data Acquisition Guide

## Data Pipeline Overview

Your complete spirometer system has **3 layers**:

### Layer 1: Hardware (Arduino/ESP32)
```
┌─────────────────────────────────────┐
│  VL53L0X Sensors (3x)              │
│  Reading ball positions (mm)        │
└──────────────┬──────────────────────┘
               │
           Serial Output
               │
   "S1:150  S2:240  S3:180"
               │
          (115200 baud)
               ▼
```

**Your Arduino code** is already doing this perfectly—reading distance values and printing them to serial.

---

### Layer 2: Data Acquisition (Python Serial Reader)
```
┌─────────────────────────────────────┐
│  spirometer_serial_reader.py        │
│  ✓ Connects to COM3                │
│  ✓ Reads sensor data for 6 seconds │
│  ✓ Converts distance → volume       │
│  ✓ Saves raw data to CSV           │
└──────────────┬──────────────────────┘
               │
        Real-time conversion:
     Distance (mm) → Volume (mL)
               │
    Using your calibration:
    Tube1: 600cc, Tube2: 900cc, Tube3: 1200cc
               │
               ▼
```

**What it does:**
- Reads Arduino's serial output in real-time
- Converts each sensor reading (distance in mm) to volume in mL
- Stores time-stamped volume data
- Saves raw data to `spirometry_PATIENT_001_TIMESTAMP.csv`

**Calibration Formula (Linear):**
```
Volume (mL) = (Distance / Max_Distance) × Max_Volume
```

Example: If sensor reads 150mm and max is 200mm, tube1 (600cc max):
```
Volume = (150 / 200) × 600 = 450 mL
```

---

### Layer 3: Analysis & Metrics (spirometer_serial_reader.py)
```
┌──────────────────────────────────────┐
│ Calculate Respiratory Metrics:       │
│                                      │
│ FVC = Maximum Volume Reached         │
│ FEV1 = Volume at 1 second           │
│ PEF = Peak Expiratory Flow (max)    │
│ FEV1/FVC = Obstructive Ratio        │
│                                      │
│ Saves to: spirometry_results_*.csv  │
└──────────────────────────────────────┘
           │
           ▼
    Results CSV:
    ┌─────────────────────────┐
    │Sensor │FVC│FEV1│FEV1/FVC│PEF│
    ├─────────────────────────┤
    │Sensor1│580│450│  77.6% │200│
    │Sensor2│850│650│  76.5% │220│
    │Sensor3│1100│780│ 70.9% │180│
    └─────────────────────────┘
```

---

## How to Use

### Step 1: Install Required Package
```powershell
pip install pyserial scipy
```

### Step 2: Check Arduino Connection
1. Connect your ESP32 to computer with USB cable
2. Open Device Manager (Windows) or `ls /dev/ttyUSB*` (Linux)
3. Find your COM port (e.g., COM3)
4. Update line 23 in `spirometer_serial_reader.py` if different:
   ```python
   SERIAL_PORT = "COM3"  # Change this if needed
   ```

### Step 3: Run a Test
```powershell
python spirometer_serial_reader.py
```

**What happens:**
1. ✓ Connects to COM3
2. ▶ Prompts: "Blow into the tubes!"
3. Reads data for 6 seconds, showing real-time volumes:
   ```
   T=0.1s | S1=45.3mL | S2=67.2mL | S3=89.1mL
   T=0.2s | S1=102.4mL | S2=156.8mL | S3=198.5mL
   ...
   ```
4. ✓ Saves two files:
   - `spirometry_PATIENT_001_20260420_143022.csv` (raw data)
   - `spirometry_results_PATIENT_001_20260420_143022.csv` (metrics)

---

## Output Files

### Raw Data CSV
```
Time (s),Sensor1_Volume (mL),Sensor2_Volume (mL),Sensor3_Volume (mL)
0.00,45.3,67.2,89.1
0.10,102.4,156.8,198.5
0.20,189.7,285.3,412.6
0.30,312.5,489.2,698.4
0.40,456.3,612.5,892.1
0.50,578.9,756.2,1050.3
0.60,612.4,834.6,1120.5
...
```

**Use this for:**
- Visualizing patient breathing patterns
- Detecting anomalies
- Training ML models (like your dashboard)

### Results CSV
```
Sensor,FVC (mL),FEV1 (mL),FEV1/FVC %,PEF (mL/s),Duration (s)
Sensor 1,612.4,489.2,79.9,245.6,6.00
Sensor 2,834.6,689.3,82.6,310.2,6.00
Sensor 3,1120.5,798.4,71.3,280.5,6.00
```

**Use this for:**
- Clinical interpretation
- Comparing against NHANES reference values
- Feeding into your ML dashboard
- Tracking patient improvement over time

---

## Connecting to Your Dashboard

Once you have test data, you can:

1. **Use raw CSV as training data** for your ML models in `combined_dashboard_v4.py`
2. **Compare results** against NHANES reference values
3. **Track patient metrics** over multiple tests

Example of next step in your workflow:
```python
# In your dashboard
test_data = pd.read_csv("spirometry_PATIENT_001_20260420_143022.csv")
metrics = pd.read_csv("spirometry_results_PATIENT_001_20260420_143022.csv")

# Compare against NHANES reference
fev1_percentile = compare_to_reference(metrics['FEV1 (mL)'], age, height, gender)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No data received" | Check USB cable, verify COM port, Arduino must print "Sensor1 OK" etc |
| "Failed to connect to COM3" | Change SERIAL_PORT variable or check Device Manager for correct port |
| Arrow volumes too low/high | Adjust `SENSOR_MAX_DISTANCE_MM` or `TUBE_MAX_VOLUMES` based on calibration |
| Serial errors | Ensure Arduino code matches format: "S1:XXX  S2:XXX  S3:XXX" |
| FEV1 = 0 | Test too short or no air flow—re-run with patient blowing harder |

---

## Customization

### Change Test Duration
Line 24 in `spirometer_serial_reader.py`:
```python
TEST_DURATION = 6  # seconds
```

### Adjust Tube Calibration
Lines 29-33:
```python
TUBE_MAX_VOLUMES = {
    'sensor1': 600,   # Change to actual tube 1 capacity
    'sensor2': 900,   # Change to actual tube 2 capacity
    'sensor3': 1200,  # Change to actual tube 3 capacity
}
```

### Change Subject ID
When running:
```python
result = run_test(subject_id="JOHN_DOE_001", output_dir=".")
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    YOUR SPIROMETER SYSTEM                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Arduino/ESP32   │
                    │  3x VL53L0X      │
                    │  Sensors         │
                    └────────┬─────────┘
                             │
                         Serial Data
                         115200 baud
                             │
                             ▼
                ┌─────────────────────────────────┐
                │ spirometer_serial_reader.py      │
                │  - Connect to COM3              │
                │  - Read for 6 seconds           │
                │  - Convert to volume (mL)       │
                │  - Detect FEV1, FVC, PEF        │
                └────────────┬────────────────────┘
                             │
                ┌────────────┴──────────────┐
                │                           │
                ▼                           ▼
    spirometry_XXXXX.csv    spirometry_results_XXXXX.csv
    (Raw Time Series)              (Metrics Table)
         │                               │
         ├──────────────┬────────────────┤
         │              │                │
         ▼              ▼                ▼
    Train Models  Visualize  Compare to NHANES
    (Dashboard)   (Excel)     Reference Values
                                        │
                                        ▼
                                  Clinical Report
```

---

## Next Steps

1. ✅ Run your Arduino code (already working)
2. ✅ Run `python spirometer_serial_reader.py` to capture test data
3. 📊 Use the CSV files in your `combined_dashboard_v4.py` to train ML models
4. 📈 Build patient comparison reports
5. 📱 Extend with web interface for multiple patients

You now have a complete real-time spirometry data acquisition pipeline! 🎉
