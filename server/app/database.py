"""
Database configuration and setup
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database URL - using SQLite for simplicity
DATABASE_URL = "sqlite:///./air_quality.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
