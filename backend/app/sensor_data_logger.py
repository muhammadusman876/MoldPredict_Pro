import serial
import csv
import time
import threading
import os
from datetime import datetime

# === CONFIGURATION ===
PORT = "COM6"       # adjust your ESP32 port
BAUD = 115200
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
CSV_FILE = os.path.join(DATA_DIR, "sensor_readings.csv")

# Create data directory structure if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# === GLOBAL VAR ===
window_status = "closed"  # default label

# === Thread for user input ===
def label_input():
    global window_status
    while True:
        key = input("Press 'o' for OPEN, 'c' for CLOSED: ").strip().lower()
        if key == 'o':
            window_status = "open"
        elif key == 'c':
            window_status = "closed"
        print(f"✅ Window status set to: {window_status}")

# Start the input thread
threading.Thread(target=label_input, daemon=True).start()

# === Serial logging ===
with serial.Serial(PORT, BAUD, timeout=1) as ser, open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["datetime", "temperature", "humidity", "window_status"])  # header
    
    print("📡 Logging started... Press 'o'/'c' anytime to label window state.")
    
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line and "," in line:
            try:
                # ESP32 still sends: millis,temp,hum
                temp, hum = line.split(",")
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([now, temp, hum, window_status])
                f.flush()
                print(now, temp, hum, window_status)
            except Exception as e:
                print("⚠️ Skipped line:", line, "| Error:", e)
