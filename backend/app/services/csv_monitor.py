"""
Thread-Safe CSV Monitor - CLEAN VERSION
Only handles CSV file monitoring, uses MoldPreventionPredictor for predictions
"""

import threading
import time
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import the dedicated mold predictor
from ..ml.mold_predictor import get_mold_predictor

logger = logging.getLogger(__name__)

class ThreadSafeCSVMonitor:
    """
    Thread-safe CSV monitoring that doesn't block the main FastAPI server
    ONLY handles CSV file monitoring - predictions done by MoldPreventionPredictor
    """
    
    def __init__(self):
        self.csv_file_path = None
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_processed_line = 0
        
        # Thread-safe storage for latest predictions
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()
        
        # Statistics
        self.total_predictions = 0
        self.alerts_triggered = 0
        self.started_at = None
    
    def start_monitoring(self, csv_file_path: str, check_interval: int = 10) -> bool:
        """
        Start CSV monitoring in a background thread
        """
        try:
            # Stop existing monitoring if active
            if self.is_monitoring:
                self.stop_monitoring()
                time.sleep(1)  # Give time to stop
            
            # Validate CSV file
            csv_path = Path(csv_file_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False
            
            # Initialize monitoring state
            self.csv_file_path = str(csv_path)
            self.is_monitoring = True
            self.last_processed_line = 0
            self.started_at = datetime.now().isoformat()
            self.total_predictions = 0
            self.alerts_triggered = 0
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(check_interval,),
                daemon=True  # Dies when main thread dies
            )
            self.monitor_thread.start()
            
            logger.info(f"Started CSV monitoring: {csv_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop CSV monitoring"""
        if not self.is_monitoring:
            return {"message": "No active monitoring to stop"}
        
        self.is_monitoring = False
        
        # Wait for thread to finish (max 2 seconds)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        stats = {
            "total_predictions": self.total_predictions,
            "alerts_triggered": self.alerts_triggered,
            "started_at": self.started_at,
            "stopped_at": datetime.now().isoformat()
        }
        
        logger.info(f"Stopped CSV monitoring. Stats: {stats}")
        return stats
    
    def _monitor_loop(self, check_interval: int):
        """
        Main monitoring loop - runs in background thread
        """
        try:
            # Get initial file size to start from end
            if self.csv_file_path and Path(self.csv_file_path).exists():
                initial_df = pd.read_csv(self.csv_file_path)
                self.last_processed_line = len(initial_df)
                logger.info(f"Starting from row {self.last_processed_line}")
            
            while self.is_monitoring:
                try:
                    self._process_new_data()
                    time.sleep(check_interval)  # Sleep between checks
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(check_interval)  # Continue despite errors
                    
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
        finally:
            logger.info("CSV monitoring loop ended")
    
    def _process_new_data(self):
        """
        Check for new CSV data and process it using MoldPreventionPredictor
        """
        try:
            if not self.csv_file_path or not Path(self.csv_file_path).exists():
                return
            
            # Read CSV file
            df = pd.read_csv(self.csv_file_path)
            current_rows = len(df)
            
            # Check for new data
            if current_rows <= self.last_processed_line:
                return  # No new data
            
            # Process new rows
            new_rows = df.iloc[self.last_processed_line:]
            logger.info(f"Processing {len(new_rows)} new rows")
            
            # Get mold predictor
            predictor = get_mold_predictor()
            
            # Process each new row
            for idx, row in new_rows.iterrows():
                try:
                    prediction_result = self._make_prediction_with_predictor(row, predictor)
                    if prediction_result:
                        self._store_prediction(prediction_result)
                        
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    continue
            
            # Update processed line count
            self.last_processed_line = current_rows
            
        except Exception as e:
            logger.error(f"Error processing CSV data: {e}")
    
    def _make_prediction_with_predictor(self, row: pd.Series, predictor) -> Optional[Dict[str, Any]]:
        """
        Use MoldPreventionPredictor to make prediction for a single row
        """
        try:
            # Extract sensor data from CSV row
            temperature = float(row['temperature'])
            humidity = float(row['humidity'])
            window_state = row.get('window_status', 'closed')  # Fixed: CSV uses 'window_status'
            heater_state = row.get('heater_state', 'off')      # Default since CSV doesn't have this
            timestamp = str(row.get('datetime', datetime.now().isoformat()))
            
            # Use the dedicated mold predictor
            current_data = {
                'indoor_temp': temperature,
                'indoor_rh': humidity,
                'window_state': window_state,
                'heater_state': heater_state
            }
            prediction = predictor.predict_rh_scenarios(current_data)
            
            if prediction:
                # Add timestamp and original sensor data
                prediction['timestamp'] = timestamp
                prediction['sensor_data'] = {
                    'temperature': temperature,
                    'humidity': humidity,
                    'window_state': window_state,
                    'heater_state': heater_state
                }
                
                return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _store_prediction(self, prediction: Dict[str, Any]):
        """
        Store prediction in thread-safe manner
        """
        try:
            # Update statistics
            self.total_predictions += 1
            if prediction.get('risk_level') == 'HIGH':
                self.alerts_triggered += 1
            
            # Store latest prediction (thread-safe)
            with self.prediction_lock:
                self.latest_prediction = prediction.copy()
                
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the latest prediction in thread-safe manner"""
        with self.prediction_lock:
            return self.latest_prediction.copy() if self.latest_prediction else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        csv_exists = False
        csv_rows = 0
        
        if self.csv_file_path:
            csv_path = Path(self.csv_file_path)
            csv_exists = csv_path.exists()
            if csv_exists:
                try:
                    df = pd.read_csv(self.csv_file_path)
                    csv_rows = len(df)
                except Exception:
                    csv_rows = 0
        
        return {
            'monitoring_active': self.is_monitoring,
            'csv_file_path': self.csv_file_path,
            'csv_exists': csv_exists,
            'csv_rows': csv_rows,
            'last_processed_line': self.last_processed_line,
            'total_predictions': self.total_predictions,
            'alerts_triggered': self.alerts_triggered,
            'started_at': self.started_at,
            'current_time': datetime.now().isoformat()
        }


# Global instance
_monitor_instance = None
_monitor_lock = threading.Lock()

def get_monitor() -> ThreadSafeCSVMonitor:
    """Get the global CSV monitor instance (thread-safe)"""
    global _monitor_instance
    
    if _monitor_instance is None:
        with _monitor_lock:
            if _monitor_instance is None:
                _monitor_instance = ThreadSafeCSVMonitor()
    
    return _monitor_instance