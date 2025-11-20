from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os
import sys
import time
import logging

# Add src to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

logger = logging.getLogger(__name__)
Base = declarative_base()

class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(255), nullable=False)
    source_type = Column(String(100), nullable=False)
    source_url = Column(Text)
    file_path = Column(Text)
    file_type = Column(String(50))
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RawData(Base):
    __tablename__ = "raw_data"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    original_content = Column(LargeBinary)
    extracted_text = Column(Text)
    metadata = Column(JSON)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataChunks(Base):
    __tablename__ = "data_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    raw_data_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text)
    chunk_embedding = Column(JSON)
    token_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class ProcessedData(Base):
    __tablename__ = "processed_data"
    
    id = Column(Integer, primary_key=True, index=True)
    raw_data_id = Column(Integer, nullable=False)
    processed_content = Column(Text)
    features = Column(JSON)
    data_type = Column(String(50))
    quality_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class VectorStore(Base):
    __tablename__ = "vector_store"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, nullable=False)
    embedding_model = Column(String(100))
    vector_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class MLModels(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100))
    model_metrics = Column(JSON)
    model_path = Column(Text)
    is_active = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalyticsReports(Base):
    __tablename__ = "analytics_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_name = Column(String(255), nullable=False)
    report_type = Column(String(100))
    report_content = Column(JSON)
    generated_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection with retry logic
def get_engine():
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("✅ Database connection successful")
            return engine
        except Exception as e:
            logger.warning(f"⚠️ Database connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ All database connection attempts failed")
                raise

def get_session():
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# Create tables
def create_tables():
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {str(e)}")
        raise
