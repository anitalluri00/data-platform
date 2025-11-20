from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os
import sys

# Add src to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

Base = declarative_base()

class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(255), nullable=False)
    source_type = Column(String(100), nullable=False)  # web, file, api, etc.
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
    content_type = Column(String(100), nullable=False)  # text, image, audio, etc.
    original_content = Column(LargeBinary)  # Store original file content
    extracted_text = Column(Text)  # Extracted text for processing
    metadata = Column(JSON)  # File metadata
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataChunks(Base):
    __tablename__ = "data_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    raw_data_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text)
    chunk_embedding = Column(JSON)  # Store vector embeddings
    token_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class ProcessedData(Base):
    __tablename__ = "processed_data"
    
    id = Column(Integer, primary_key=True, index=True)
    raw_data_id = Column(Integer, nullable=False)
    processed_content = Column(Text)
    features = Column(JSON)  # Extracted features
    data_type = Column(String(50))  # structured, unstructured, etc.
    quality_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class VectorStore(Base):
    __tablename__ = "vector_store"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, nullable=False)
    embedding_model = Column(String(100))
    vector_data = Column(JSON)  # Store vector embeddings
    created_at = Column(DateTime, default=datetime.utcnow)

class MLModels(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100))  # classification, regression, clustering
    model_metrics = Column(JSON)
    model_path = Column(Text)
    is_active = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalyticsReports(Base):
    __tablename__ = "analytics_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_name = Column(String(255), nullable=False)
    report_type = Column(String(100))  # EDA, summary, insights
    report_content = Column(JSON)
    generated_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection functions
def get_engine():
    return create_engine(settings.DATABASE_URL)

def get_session():
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# Create tables
def create_tables():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)