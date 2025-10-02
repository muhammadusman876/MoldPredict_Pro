"""
Mold Prevention Model Loader
Handles XGBoost model loading and real-time CSV data processing for mold risk prediction
"""

import os
import joblib
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class MoldPreventionPredictor:
    """
    XGBoost-based mold prevention predictor
    Loads your trained model and processes real-time DHT22 sensor data
    """
    
    def __init__(self, model_dir: str = "data/models"):
        # Resolve path relative to the app directory
        if not os.path.isabs(model_dir):
            app_dir = Path(__file__).parent.parent  # Go up from ml/ to app/
            self.model_dir = app_dir / model_dir
        else:
            self.model_dir = Path(model_dir)
        self.model = None
        self.window_encoder = None
        self.heater_encoder = None
        self.is_loaded = False
        
        # Model performance specs
        self.target_r2 = 0.90
        self.target_rmse = 3.0
        
        # Mold risk thresholds
        self.safe_threshold = 60.0      # RH < 60% = Safe
        self.warning_threshold = 65.0   # 60% <= RH < 65% = Warning
        self.critical_threshold = 65.0  # RH >= 65% = Critical mold risk
        
        # Mock outdoor conditions (since we only have indoor sensors)
        self.mock_outdoor_temp = 10.0   # Default outdoor temperature
        self.mock_outdoor_rh = 80.0     # Default outdoor humidity
        self.mock_activity = 1.0        # Default activity level
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """Load XGBoost model and encoders from your training session"""
        try:
            model_files = {
                'model': 'humidity_predictor_model.joblib',
                'window_encoder': 'window_encoder.pkl', 
                'heater_encoder': 'heater_encoder.pkl'
            }
            
            # Check if model files exist
            missing_files = []
            for name, filename in model_files.items():
                filepath = self.model_dir / filename
                if not filepath.exists():
                    missing_files.append(filename)
            
            if missing_files:
                logger.warning(f"Missing model files: {missing_files}")
                return False
            
            # Load XGBoost model
            model_path = self.model_dir / model_files['model']
            self.model = joblib.load(model_path)
            logger.info(f"Loaded XGBoost model from {model_path}")
            
            # Load label encoders
            with open(self.model_dir / model_files['window_encoder'], 'rb') as f:
                self.window_encoder = pickle.load(f)
            
            with open(self.model_dir / model_files['heater_encoder'], 'rb') as f:
                self.heater_encoder = pickle.load(f)
            
            self.is_loaded = True
            logger.info("Mold prevention model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load mold prevention model: {e}")
            self.is_loaded = False
            return False
    
    def calculate_absolute_humidity(self, temp_celsius: float, rh_percent: float) -> float:
        """
        SAFE: Calculate Absolute Humidity (AH) in g/m³
        Uses Magnus formula - FIXED to prevent hanging
        """
        try:
            # Ensure simple float inputs (prevent numpy array issues)
            temp = float(temp_celsius)
            rh = float(rh_percent)
            
            # Validate inputs to prevent infinite loops
            if temp < -50 or temp > 100:
                logger.warning(f"Invalid temperature {temp}, using default")
                temp = 20.0
            if rh < 0 or rh > 100:
                logger.warning(f"Invalid RH {rh}, using default") 
                rh = 50.0
            
            # Magnus formula constants
            a = 17.27
            b = 237.7
            
            # Use math.exp instead of np.exp to avoid potential numpy issues
            import math
            es = 6.112 * math.exp((a * temp) / (b + temp))
            
            # Actual vapor pressure
            e = (rh / 100.0) * es
            
            # Absolute humidity in g/m³
            ah = (2.16679 * e) / (273.15 + temp)
            
            # Sanity check result
            if ah < 0 or ah > 50:
                logger.warning(f"Invalid AH result {ah}, using default")
                return 5.0
                
            return round(float(ah), 2)
            
        except Exception as e:
            logger.error(f"Error calculating absolute humidity: {e}")
            return 5.0  # Safe default
    
    def calculate_delta_ah(self, indoor_temp: float, indoor_rh: float, 
                          outdoor_temp: float, outdoor_rh: float) -> float:
        """
        Calculate Delta_AH = AH_indoor - AH_outdoor
        Key physics feature for window effectiveness prediction
        """
        ah_indoor = self.calculate_absolute_humidity(indoor_temp, indoor_rh)
        ah_outdoor = self.calculate_absolute_humidity(outdoor_temp, outdoor_rh)
        
        delta_ah = ah_indoor - ah_outdoor
        logger.debug(f"Delta_AH: {delta_ah:.2f} g/m³ (Indoor: {ah_indoor}, Outdoor: {ah_outdoor})")
        
        return round(delta_ah, 2)
    
    def read_csv_data(self, csv_path: str, time_column: str = 'timestamp') -> pd.DataFrame:
        """
        Read real-time sensor data from CSV file
        Expected format: timestamp, indoor_temp, indoor_rh, outdoor_temp, outdoor_rh, activity
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Convert timestamp column
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.sort_values(time_column)
            
            logger.info(f"Loaded {len(df)} records from {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV data: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, current_data: Dict[str, Any], 
                        window_action: str = 'closed', 
                        heater_action: str = 'off',
                        historical_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prepare features for XGBoost prediction matching your training format
        """
        try:
            features = {}
            
            # Current conditions
            indoor_temp = current_data.get('indoor_temp', 22.0)
            indoor_rh = current_data.get('indoor_rh', 50.0)
            outdoor_temp = current_data.get('outdoor_temp', 20.0)
            outdoor_rh = current_data.get('outdoor_rh', 60.0)
            activity = current_data.get('activity', 0)  # 0=Absent, 1=Present
            
            # 1. Delta_AH (key physics feature)
            features['Delta_AH'] = self.calculate_delta_ah(
                indoor_temp, indoor_rh, outdoor_temp, outdoor_rh
            )
            
            # 2. Lagged variables (15 minutes ago)
            try:
                if historical_df is not None and len(historical_df) > 450:  # Need at least 450 rows for 15min lag
                    # Use simple row-based lag (assuming 2-second intervals like training)
                    lag_rows = 450  # 15 minutes * 60 seconds / 2 seconds per row
                    lag_idx = len(historical_df) - lag_rows - 1
                    
                    if lag_idx >= 0:
                        lag_row = historical_df.iloc[lag_idx]
                        features['RH_15min_ago'] = float(lag_row['indoor_rh'])
                        features['Temp_15min_ago'] = float(lag_row['indoor_temp'])
                    else:
                        features['RH_15min_ago'] = indoor_rh
                        features['Temp_15min_ago'] = indoor_temp
                else:
                    # Use current values if insufficient historical data
                    features['RH_15min_ago'] = indoor_rh
                    features['Temp_15min_ago'] = indoor_temp
            except Exception as e:
                logger.warning(f"Error processing lagged features: {e}")
                features['RH_15min_ago'] = indoor_rh
                features['Temp_15min_ago'] = indoor_temp
            
            # 3. Activity label
            features['Activity_Label'] = activity
            
            # 4. Action labels (simplified - not used in prediction, just for logging)
            # Based on your training, these aren't input features, just for scenario naming
            window_encoded = 0 if window_action == 'closed' else 1  
            heater_encoded = 0 if heater_action == 'off' else 1            # Convert to array in EXACT order matching your training data
            # Based on your Jupyter code: ['Temp_in', 'RH_in', 'Temp_out', 'RH_out', 'Delta_AH', 'RH_15min_ago', 'Temp_15min_ago', 'Activity']
            feature_array = [
                float(indoor_temp),                    # Temp_in
                float(indoor_rh),                      # RH_in  
                float(outdoor_temp),                   # Temp_out
                float(outdoor_rh),                     # RH_out
                float(features['Delta_AH']),           # Delta_AH
                float(features['RH_15min_ago']),       # RH_15min_ago
                float(features['Temp_15min_ago']),     # Temp_15min_ago
                float(activity)                        # Activity
            ]
            
            logger.info(f"XGBoost features (8 total): {feature_array}")
            
            # Ensure exactly 8 features
            if len(feature_array) != 8:
                logger.error(f"Feature count mismatch! Expected 8, got {len(feature_array)}")
                # Pad or truncate to 8 features
                while len(feature_array) < 8:
                    feature_array.append(0.0)
                feature_array = feature_array[:8]
            
            return np.array(feature_array).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return safe 8-feature array matching training format
            return np.array([22.0, 50.0, 10.0, 80.0, 0.0, 50.0, 22.0, 1.0]).reshape(1, -1)
    
    def predict_rh_scenarios(self, current_data: Dict[str, Any], 
                            historical_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        SIMPLIFIED: Predict RH 15 minutes ahead (single scenario to avoid hanging)
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded. Place your trained model files in models/ folder',
                'current_rh': current_data.get('indoor_rh', 0),
                'recommendation': {'predicted_rh': 0, 'risk_level': 'unknown', 'actions': 'Model not loaded'}
            }
        
        try:
            # ULTRA-SIMPLIFIED: Skip complex feature preparation, use direct approach
            if self.model is None:
                return {'error': 'Model not loaded properly'}
            
            # Direct feature array (like our successful test)
            indoor_temp = float(current_data.get('indoor_temp', 22.0))
            indoor_rh = float(current_data.get('indoor_rh', 50.0))
            
            # Use configurable mock values for missing features
            outdoor_temp = float(current_data.get('outdoor_temp', self.mock_outdoor_temp))
            outdoor_rh = float(current_data.get('outdoor_rh', self.mock_outdoor_rh))
            activity = float(current_data.get('activity', self.mock_activity))
            
            # Calculate Delta_AH directly (avoid method call that might hang)
            import math
            def calc_ah(temp, rh):
                es = 6.112 * math.exp((17.27 * temp) / (237.7 + temp))
                e = (rh / 100.0) * es
                return (2.16679 * e) / (273.15 + temp)
            
            delta_ah = calc_ah(indoor_temp, indoor_rh) - calc_ah(outdoor_temp, outdoor_rh)
            
            # Use current values for lagged features (simplified)
            rh_15min_ago = indoor_rh
            temp_15min_ago = indoor_temp
            
            # Build feature array directly (no complex processing)
            features = np.array([[
                indoor_temp, indoor_rh, outdoor_temp, outdoor_rh,
                delta_ah, rh_15min_ago, temp_15min_ago, activity
            ]])
            
            # Make single prediction
            predicted_rh = float(self.model.predict(features)[0])
            current_rh = float(current_data.get('indoor_rh', 0))
            
            # Assess mold risk
            if predicted_rh < 60.0:
                risk_level = 'safe'
                actions = 'No action needed - conditions are safe'
            elif predicted_rh < 65.0:
                risk_level = 'warning' 
                actions = 'Consider opening windows for ventilation'
            else:
                risk_level = 'critical'
                actions = 'IMMEDIATE ACTION: Open windows AND turn on heater/dehumidifier'
            
            # Simple recommendation
            recommendation = {
                'predicted_rh': round(predicted_rh, 1),
                'risk_level': risk_level,
                'actions': actions,
                'confidence': 'high' if abs(predicted_rh - current_rh) < 5 else 'medium'
            }
            
            return {
                'status': 'success',
                'current_rh': current_rh,
                'prediction': predicted_rh,
                'recommendation': recommendation,
                'timestamp': current_data.get('timestamp', datetime.now().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'scenarios': {},
                'recommendation': None
            }
    
    def _estimate_energy_cost(self, window_action: str, heater_action: str) -> str:
        """Estimate relative energy cost of actions"""
        if window_action == 'open' and heater_action == 'off':
            return 'zero'  # Natural ventilation is free
        elif window_action == 'closed' and heater_action == 'off':
            return 'zero'  # No action is free
        elif heater_action == 'on':
            return 'high'  # Heating costs energy
        else:
            return 'low'
    
    def _select_best_action(self, scenarios: Dict) -> Dict:
        """Select best action based on RH reduction and energy efficiency"""
        # Filter scenarios that keep RH below critical threshold
        safe_scenarios = {k: v for k, v in scenarios.items() 
                         if v['predicted_rh'] < self.critical_threshold}
        
        if not safe_scenarios:
            # If no scenario is safe, choose the one with lowest RH
            best_key = min(scenarios.keys(), key=lambda k: scenarios[k]['predicted_rh'])
        else:
            # Among safe scenarios, prefer energy-efficient ones
            energy_priority = {'zero': 0, 'low': 1, 'high': 2}
            best_key = min(safe_scenarios.keys(), 
                          key=lambda k: (safe_scenarios[k]['predicted_rh'],
                                       energy_priority.get(safe_scenarios[k]['energy_cost'], 3)))
        
        return scenarios[best_key]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded mold prevention model"""
        return {
            'is_loaded': self.is_loaded,
            'model_type': 'XGBoost Regressor',
            'target_performance': {
                'r_squared': f'>= {self.target_r2}',
                'rmse': f'<= {self.target_rmse}% RH'
            },
            'features': ['Delta_AH', 'RH_15min_ago', 'Temp_15min_ago', 
                        'Activity_Label', 'Window_Status', 'Heater_Status'],
            'thresholds': {
                'safe': f'< {self.safe_threshold}% RH',
                'warning': f'{self.safe_threshold}-{self.critical_threshold}% RH', 
                'critical': f'>= {self.critical_threshold}% RH'
            },
            'actions': ['window_open', 'window_closed', 'heater_on', 'heater_off'],
            'prediction_horizon': '15 minutes ahead'
        }

# Global instance
_mold_predictor = None

def get_mold_predictor() -> MoldPreventionPredictor:
    """Get global mold prevention predictor instance"""
    global _mold_predictor
    if _mold_predictor is None:
        _mold_predictor = MoldPreventionPredictor()
    return _mold_predictor

def reload_mold_model() -> bool:
    """Reload the mold prevention model"""
    global _mold_predictor
    _mold_predictor = MoldPreventionPredictor()
    return _mold_predictor.is_loaded