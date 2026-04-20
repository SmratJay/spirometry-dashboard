"""
Real-time Spirometer Data Acquisition & Processing
Reads distance values from Arduino/ESP32 and converts to respiratory metrics
"""

import serial
import csv
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import threading

# ============================================================================
# CONFIGURATION - ADJUST THESE FOR YOUR HARDWARE
# ============================================================================

SERIAL_PORT = "COM3"  # Change if different
BAUD_RATE = 115200
TEST_DURATION = 6  # seconds (for FEV1 measurement)
SAMPLING_RATE = 10  # Hz (read faster than 500ms intervals for better data)

# Tube Calibration: Adjust based on your tube dimensions
# These are maximum volumes per tube (in mL)
TUBE_MAX_VOLUMES = {
    'sensor1': 600,   # Tube 1: 600cc max
    'sensor2': 900,   # Tube 2: 900cc max
    'sensor3': 1200,  # Tube 3: 1200cc max
}

# Maximum sensor distance (when ball is at bottom, full volume)
# Adjust this based on your VL53L0X sensor range
SENSOR_MAX_DISTANCE_MM = 200  # Typical range for VL53L0X

# ============================================================================
# CALIBRATION FUNCTIONS
# ============================================================================

def distance_to_volume(distance_mm, sensor_id, max_distance=SENSOR_MAX_DISTANCE_MM):
    """
    Convert sensor distance (mm) to volume (mL)
    
    Assumption: Sensor reads 0mm when ball is at top (no volume)
                Sensor reads max when ball is at bottom (full volume)
    
    This is LINEAR calibration. If your tubes are non-linear, adjust accordingly.
    """
    sensor_key = f'sensor{sensor_id}'
    max_volume = TUBE_MAX_VOLUMES.get(sensor_key, 600)
    
    # Linear conversion: volume = (distance / max_distance) * max_volume
    volume = (distance_mm / max_distance) * max_volume
    
    # Cap at max volume
    return min(max(volume, 0), max_volume)


def calculate_respiratory_metrics(timestamps, volumes, sensor_id):
    """
    Calculate FEV1, FVC, PEF from volume-time curve
    
    Returns:
        dict with metrics: FEV1, FVC, PEF, FEV1/FVC ratio, etc.
    """
    
    if len(volumes) < 3:
        return None
    
    # Find maximum volume (FVC - Forced Vital Capacity)
    fvc = np.max(volumes)
    fvc_idx = np.argmax(volumes)
    fvc_time = timestamps[fvc_idx]
    
    # Find volume at 1 second (FEV1)
    fev1_idx = np.argmin(np.abs(np.array(timestamps) - (timestamps[0] + 1.0)))
    fev1 = volumes[fev1_idx] if fev1_idx < len(volumes) else 0
    
    # Peak Expiratory Flow (PEF) - max slope
    if len(volumes) > 1:
        flow_rates = np.diff(volumes) / np.diff(timestamps)
        pef = np.max(flow_rates) if len(flow_rates) > 0 else 0
    else:
        pef = 0
    
    # Calculate ratios
    fev1_fvc_ratio = (fev1 / fvc) if fvc > 0 else 0
    
    return {
        'sensor_id': sensor_id,
        'fvc': round(fvc, 2),
        'fev1': round(fev1, 2),
        'fev1_fvc_ratio': round(fev1_fvc_ratio, 4),
        'pef': round(pef, 2),
        'test_duration': round(timestamps[-1] - timestamps[0], 2)
    }


# ============================================================================
# SERIAL COMMUNICATION
# ============================================================================

class SpirometryReader:
    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE):
        self.port = port
        self.baud = baud
        self.ser = None
        self.data = {'timestamps': [], 'sensor1': [], 'sensor2': [], 'sensor3': []}
        self.running = False
        
    def connect(self):
        """Connect to serial port"""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"✓ Connected to {self.port} at {self.baud} baud")
            time.sleep(2)  # Wait for Arduino to initialize
            return True
        except Exception as e:
            print(f"✗ Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Safely disconnect from serial port"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✓ Disconnected from serial port")
    
    def read_test(self, duration=TEST_DURATION):
        """
        Read sensor data for the specified duration
        
        Returns:
            dict with timestamps and sensor readings
        """
        if not self.ser or not self.ser.is_open:
            if not self.connect():
                return None
        
        self.data = {'timestamps': [], 'sensor1': [], 'sensor2': [], 'sensor3': []}
        start_time = time.time()
        
        print(f"\n▶ Starting {duration}s test... Blow into the tubes!")
        
        while (time.time() - start_time) < duration:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8').strip()
                    
                    # Expected format: "S1:123  S2:234  S3:345"
                    if "S1:" in line:
                        parts = line.split()
                        
                        # Parse sensor values
                        s1_val = int(parts[0].split(':')[1]) if len(parts) > 0 else 0
                        s2_val = int(parts[1].split(':')[1]) if len(parts) > 1 else 0
                        s3_val = int(parts[2].split(':')[1]) if len(parts) > 2 else 0
                        
                        elapsed = time.time() - start_time
                        
                        self.data['timestamps'].append(elapsed)
                        self.data['sensor1'].append(distance_to_volume(s1_val, 1))
                        self.data['sensor2'].append(distance_to_volume(s2_val, 2))
                        self.data['sensor3'].append(distance_to_volume(s3_val, 3))
                        
                        # Real-time feedback
                        print(f"  T={elapsed:.1f}s | S1={self.data['sensor1'][-1]:.1f}mL | "
                              f"S2={self.data['sensor2'][-1]:.1f}mL | S3={self.data['sensor3'][-1]:.1f}mL")
                
                time.sleep(1 / SAMPLING_RATE)  # Sample at SAMPLING_RATE Hz
                
            except Exception as e:
                print(f"✗ Error reading serial: {e}")
                break
        
        print("✓ Test complete!\n")
        return self.data
    
    def save_to_csv(self, test_data, subject_id="TEST", output_dir="."):
        """
        Save test data to CSV file
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spirometry_{subject_id}_{timestamp_str}.csv"
        filepath = Path(output_dir) / filename
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['Time (s)', 'Sensor1_Volume (mL)', 'Sensor2_Volume (mL)', 'Sensor3_Volume (mL)'])
                
                # Data rows
                for i in range(len(test_data['timestamps'])):
                    writer.writerow([
                        f"{test_data['timestamps'][i]:.2f}",
                        f"{test_data['sensor1'][i]:.2f}",
                        f"{test_data['sensor2'][i]:.2f}",
                        f"{test_data['sensor3'][i]:.2f}"
                    ])
            
            print(f"✓ Saved to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"✗ Failed to save CSV: {e}")
            return None


# ============================================================================
# ANALYSIS & RESULTS
# ============================================================================

def analyze_test(test_data):
    """Analyze test data and calculate respiratory metrics"""
    
    results = {}
    
    for sensor_id in [1, 2, 3]:
        sensor_key = f'sensor{sensor_id}'
        volumes = test_data[sensor_key]
        timestamps = test_data['timestamps']
        
        metrics = calculate_respiratory_metrics(timestamps, volumes, sensor_id)
        if metrics:
            results[sensor_key] = metrics
            print(f"\nSensor {sensor_id} Results:")
            print(f"  FVC (Forced Vital Capacity): {metrics['fvc']} mL")
            print(f"  FEV1 (Forced Expiration in 1s): {metrics['fev1']} mL")
            print(f"  FEV1/FVC Ratio: {metrics['fev1_fvc_ratio'] * 100:.1f}%")
            print(f"  PEF (Peak Expiratory Flow): {metrics['pef']} mL/s")
    
    return results


def save_results_to_csv(results, subject_id="TEST", output_dir="."):
    """Save calculated metrics to a results CSV"""
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spirometry_results_{subject_id}_{timestamp_str}.csv"
    filepath = Path(output_dir) / filename
    
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sensor', 'FVC (mL)', 'FEV1 (mL)', 'FEV1/FVC %', 'PEF (mL/s)', 'Duration (s)'])
            
            for sensor_key, metrics in results.items():
                writer.writerow([
                    f"Sensor {metrics['sensor_id']}",
                    metrics['fvc'],
                    metrics['fev1'],
                    f"{metrics['fev1_fvc_ratio'] * 100:.1f}",
                    metrics['pef'],
                    metrics['test_duration']
                ])
        
        print(f"✓ Results saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        return None


# ============================================================================
# MAIN TESTING INTERFACE
# ============================================================================

def run_test(subject_id="TEST", output_dir="."):
    """
    Complete test workflow:
    1. Connect to serial
    2. Read data for 6 seconds
    3. Analyze and calculate metrics
    4. Save both raw data and results
    """
    
    reader = SpirometryReader()
    
    if not reader.connect():
        return None
    
    # Run the test
    test_data = reader.read_test(duration=TEST_DURATION)
    reader.disconnect()
    
    if not test_data or len(test_data['timestamps']) == 0:
        print("✗ No data received. Check Arduino connection.")
        return None
    
    # Save raw data
    print("\n" + "="*60)
    print("SAVING DATA AND ANALYZING...")
    print("="*60)
    
    data_file = reader.save_to_csv(test_data, subject_id, output_dir)
    
    # Analyze results
    results = analyze_test(test_data)
    
    # Save results
    results_file = save_results_to_csv(results, subject_id, output_dir)
    
    return {
        'data_file': data_file,
        'results_file': results_file,
        'metrics': results
    }


if __name__ == "__main__":
    # Example: Run a test
    # Change "PATIENT_001" to your subject ID
    result = run_test(subject_id="PATIENT_001", output_dir=".")
    
    if result:
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print(f"Raw data: {result['data_file']}")
        print(f"Results: {result['results_file']}")
        print("="*60)
