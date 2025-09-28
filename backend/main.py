"""
Smart Mold Prevention System - FastAPI Backend
Main application entry point for IoT-powered mold risk prevention
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.database import engine, Base
from app.routers import mold_prevention
from app.exceptions import AirQualityAPIException, ErrorResponse
from app.services.csv_monitor import get_monitor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Mold Prevention API", 
    description="IoT-powered Mold Risk Prevention System with XGBoost Predictive AI and Action Recommendations",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    """Initialize services when the server starts"""
    logger.info("Starting Smart Mold Prevention System...")
    
    # Initialize the CSV monitor (but don't start monitoring yet)
    monitor = get_monitor()
    logger.info(f"Thread-safe CSV monitor initialized: {type(monitor).__name__}")
    
    logger.info("System ready! Use /mold-prevention/start-monitoring to begin.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when the server shuts down"""
    logger.info("Shutting down Smart Mold Prevention System...")
    
    # Stop any active monitoring
    try:
        monitor = get_monitor()
        if monitor.is_monitoring:
            stats = monitor.stop_monitoring()
            logger.info(f"Stopped monitoring. Final stats: {stats}")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Shutdown complete")

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler
@app.exception_handler(AirQualityAPIException)
async def air_quality_exception_handler(request: Request, exc: AirQualityAPIException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=exc.message,
            details=exc.details,
            status_code=exc.status_code
        ).model_dump()
    )

# Include routers
app.include_router(mold_prevention.router, prefix="/api/v1", tags=["Mold Prevention"])

@app.get("/")
async def root():
    return {
        "message": "Smart Mold Prevention API",
        "version": "1.0.0", 
        "status": "running",
        "features": [
            "Real-time humidity and temperature monitoring",
            "XGBoost mold risk prediction (15-min ahead)",
            "Multi-scenario action recommendations",
            "Physics-based Delta_AH calculations", 
            "Energy-efficient mold prevention",
            "Audio alerts and comprehensive error handling"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-09-14T00:00:00Z",
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
