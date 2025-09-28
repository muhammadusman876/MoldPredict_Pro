"""
Configuration settings for the Air Quality Monitor backend
"""

import os
from typing import Optional

class Settings:
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./air_quality.db")
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Smart Air Quality Monitor"
    VERSION: str = "1.0.0"
    
    # CORS settings
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",  # React frontend
        "http://localhost:3001",  # Alternative frontend port
        "http://127.0.0.1:3000",
    ]
    
    # Synthetic data settings
    DEFAULT_LOCATION: str = "living_room"
    MAX_SYNTHETIC_READINGS: int = 1000
    
    # Air quality thresholds
    CO2_THRESHOLDS = {
        "excellent": 400,
        "good": 800,
        "moderate": 1200,
        "poor": 2000
    }
    
    TEMP_THRESHOLDS = {
        "optimal_min": 20,
        "optimal_max": 24,
        "comfortable_min": 18,
        "comfortable_max": 26
    }
    
    HUMIDITY_THRESHOLDS = {
        "optimal_min": 40,
        "optimal_max": 60,
        "comfortable_min": 30,
        "comfortable_max": 70
    }

settings = Settings()
