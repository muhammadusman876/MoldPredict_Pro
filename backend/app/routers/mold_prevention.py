from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

from ..services.csv_monitor import get_monitor
from ..ml.mold_predictor import get_mold_predictor, reload_mold_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mold-prevention", tags=["Mold Prevention"])

class MonitoringConfig(BaseModel):
    csv_file_path: str = Field(..., description="Path to CSV file")
    check_interval: int = Field(default=10, description="Check interval seconds")

@router.post("/start-monitoring")
async def start_monitoring(config: MonitoringConfig):
    """Start thread-safe CSV monitoring for mold risk prediction"""
    try:
        monitor = get_monitor()
        success = monitor.start_monitoring(
            csv_file_path=config.csv_file_path,
            check_interval=config.check_interval
        )
        
        return {
            "status": "success" if success else "info",
            "message": f"Started monitoring {config.csv_file_path}" if success else "Monitoring restarted"
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-monitoring")
async def stop_monitoring():
    """Stop thread-safe CSV monitoring"""
    try:
        monitor = get_monitor()
        stats = monitor.stop_monitoring()
        return {
            "status": "success",
            "message": "Monitoring stopped",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Get current monitoring status"""
    try:
        monitor = get_monitor()
        status = monitor.get_status()
        return {
            "status": "success",
            "monitoring": status
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest-prediction")
async def get_latest_prediction():
    """Get the most recent mold risk prediction"""
    try:
        monitor = get_monitor()
        prediction = monitor.get_latest_prediction()
        
        if prediction:
            return {
                "status": "success",
                "prediction": prediction
            }
        else:
            return {
                "status": "no_predictions",
                "message": "No predictions available"
            }
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Check system health"""
    try:
        monitor = get_monitor()
        predictor = get_mold_predictor()
        
        return {
            "status": "healthy",
            "monitor_available": monitor is not None,
            "model_loaded": predictor.model is not None,
            "thread_safe": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}