"""
Pagination utilities for API responses
"""

from pydantic import BaseModel, Field
from typing import List, TypeVar, Generic, Optional
from math import ceil

T = TypeVar('T')

class PaginationParams(BaseModel):
    """Pagination parameters for API requests"""
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    limit: int = Field(default=50, ge=1, le=1000, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database query"""
        return (self.page - 1) * self.limit

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response model"""
    items: List[T]
    total: int
    page: int
    limit: int
    pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        limit: int
    ) -> "PaginatedResponse[T]":
        """Create a paginated response"""
        pages = ceil(total / limit) if total > 0 else 1
        
        return cls(
            items=items,
            total=total,
            page=page,
            limit=limit,
            pages=pages,
            has_next=page < pages,
            has_previous=page > 1
        )

class ReadingsPaginatedResponse(PaginatedResponse):
    """Specific paginated response for air quality readings"""
    summary: Optional[dict] = Field(None, description="Summary statistics for the current page")
    
    @classmethod
    def create_with_summary(
        cls,
        items: List,
        total: int,
        page: int,
        limit: int,
        summary: Optional[dict] = None
    ) -> "ReadingsPaginatedResponse":
        """Create a paginated response with summary statistics"""
        pages = ceil(total / limit) if total > 0 else 1
        
        return cls(
            items=items,
            total=total,
            page=page,
            limit=limit,
            pages=pages,
            has_next=page < pages,
            has_previous=page > 1,
            summary=summary
        )
