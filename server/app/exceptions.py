"""
Custom exception handlers and error models for the Air Quality API
"""

from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    status_code: int

class AirQualityAPIException(HTTPException):
    """Base exception for Air Quality API"""
    def __init__(
        self, 
        status_code: int, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(status_code=status_code, detail=message)

class DatabaseException(AirQualityAPIException):
    """Database operation failed"""
    def __init__(self, message: str = "Database operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            details=details
        )

class ValidationException(AirQualityAPIException):
    """Data validation failed"""
    def __init__(self, message: str = "Data validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            details=details
        )

class NotFoundException(AirQualityAPIException):
    """Resource not found"""
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            details=details
        )

class RateLimitException(AirQualityAPIException):
    """Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message=message,
            details=details
        )

def log_and_raise_db_error(operation: str, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log database error and raise appropriate exception"""
    error_details = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    if context:
        error_details.update(context)
    
    logger.error(f"Database error in {operation}: {error}", extra=error_details)
    
    raise DatabaseException(
        message=f"Failed to {operation}",
        details=error_details
    )

def validate_reading_data(data: Dict[str, Any]) -> None:
    """Validate air quality reading data"""
    required_fields = ["co2_ppm", "temperature_celsius", "humidity_percent"]
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValidationException(
            message="Missing required fields",
            details={"missing_fields": missing_fields}
        )
    
    # Validate ranges
    if not (0 <= data["co2_ppm"] <= 50000):
        raise ValidationException(
            message="CO2 value out of valid range",
            details={"co2_ppm": data["co2_ppm"], "valid_range": "0-50000 ppm"}
        )
    
    if not (-50 <= data["temperature_celsius"] <= 100):
        raise ValidationException(
            message="Temperature value out of valid range",
            details={"temperature_celsius": data["temperature_celsius"], "valid_range": "-50 to 100°C"}
        )
    
    if not (0 <= data["humidity_percent"] <= 100):
        raise ValidationException(
            message="Humidity value out of valid range",
            details={"humidity_percent": data["humidity_percent"], "valid_range": "0-100%"}
        )
