# XGBoost Mold Prevention Model Setup

## 📋 Required Model Files

Place your trained XGBoost model files in this directory:

```
backend/app/data/models/
├── humidity_predictor_model.joblib    # Main XGBoost model
├── window_encoder.pkl                 # LabelEncoder for window actions
└── heater_encoder.pkl                 # LabelEncoder for heater actions
```

## 🧪 Model Training (Google Colab)

Your model should be trained with these features:

### Input Features (X):

1. **Delta_AH**: Absolute Humidity difference (AH_indoor - AH_outdoor) [g/m³]
2. **RH_15min_ago**: Relative Humidity 15 minutes ago [%]
3. **Temp_15min_ago**: Temperature 15 minutes ago [°C]
4. **Activity_Label**: Occupancy status (0=Absent, 1=Present)
5. **Window_Status**: Window action (0=Closed, 1=Open) - encoded
6. **Heater_Status**: Heater action (0=Off, 1=On) - encoded

### Target Variable (Y):

- **RH_future**: Relative Humidity 15 minutes ahead [%]

### Performance Requirements:

- **R² ≥ 0.90** (coefficient of determination)
- **RMSE ≤ 3.0%** (root mean square error in RH percentage)

## 💾 Model Saving (Colab)

```python
import joblib
import pickle

# Save XGBoost model
joblib.dump(xgb_model, 'humidity_predictor_model.joblib')

# Save label encoders
with open('window_encoder.pkl', 'wb') as f:
    pickle.dump(window_label_encoder, f)

with open('heater_encoder.pkl', 'wb') as f:
    pickle.dump(heater_label_encoder, f)

# Download files from Colab
from google.colab import files
files.download('humidity_predictor_model.joblib')
files.download('window_encoder.pkl')
files.download('heater_encoder.pkl')
```

## 🔄 Model Loading Process

The system automatically:

1. **Scans models/ directory** for required files
2. **Loads XGBoost model** using joblib
3. **Loads label encoders** using pickle
4. **Validates model** is ready for predictions
5. **Logs status** to confirm successful loading

## 📊 Prediction Process

### Multi-Scenario Prediction:

For each sensor reading, the system predicts RH for 4 scenarios:

- **Scenario A**: Window Closed + Heater Off (baseline)
- **Scenario B**: Window Open + Heater Off (natural ventilation)
- **Scenario C**: Window Closed + Heater On (heating only)
- **Scenario D**: Window Open + Heater On (ventilation + heating)

### Action Selection:

1. **Safety First**: Prefer actions that keep RH < 65% (mold threshold)
2. **Energy Efficiency**: Prefer natural ventilation over mechanical solutions
3. **Comfort Balance**: Consider temperature impact of actions

## 🧮 Physics Calculations

### Absolute Humidity Formula:

```python
def calculate_absolute_humidity(temp_celsius, rh_percent):
    # Magnus formula constants
    a = 17.27
    b = 237.7

    # Saturation vapor pressure
    es = 6.112 * exp((a * temp_celsius) / (b + temp_celsius))

    # Actual vapor pressure
    e = (rh_percent / 100.0) * es

    # Absolute humidity [g/m³]
    ah = (2.16679 * e) / (273.15 + temp_celsius)

    return ah

# Delta_AH calculation
delta_ah = calculate_absolute_humidity(indoor_temp, indoor_rh) - \
           calculate_absolute_humidity(outdoor_temp, outdoor_rh)
```

## 🔍 Model Validation

Check model performance:

```bash
# Start backend
cd backend && python -m uvicorn main:app --reload

# Check model status
curl http://localhost:8000/api/v1/mold-prevention/model-info

# Test prediction
curl -X POST http://localhost:8000/api/v1/mold-prevention/predict \
  -H "Content-Type: application/json" \
  -d '{
    "indoor_temp": 22.5,
    "indoor_rh": 58.0,
    "outdoor_temp": 18.0,
    "outdoor_rh": 72.0,
    "activity": 1
  }'
```

## 📁 File Status

- [ ] `humidity_predictor_model.joblib` - **MISSING** (place your XGBoost model here)
- [ ] `window_encoder.pkl` - **MISSING** (place your window encoder here)
- [ ] `heater_encoder.pkl` - **MISSING** (place your heater encoder here)

## 🚨 Troubleshooting

**"Model not loaded" error?**

1. Check all 3 files are in `backend/app/data/models/` directory
2. Verify file names match exactly (case-sensitive)
3. Ensure files are valid joblib/pickle format
4. Check backend logs for specific error messages

**Prediction errors?**

1. Verify input data has all required fields
2. Check temperature/humidity values are reasonable
3. Ensure activity is 0 or 1
4. Check timestamp format if provided

**Performance issues?**

1. Monitor R² and RMSE metrics
2. Retrain model with more representative data
3. Add more physics-based features (wind speed, pressure)
4. Increase training data diversity (different seasons/conditions)
