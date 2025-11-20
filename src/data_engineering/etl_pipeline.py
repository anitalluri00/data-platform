import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from database.operations import DatabaseOperations
from data_science.preprocessing import DataPreprocessor
from data_science.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self):
        self.db_ops = DatabaseOperations()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
    
    def run_pipeline(self, source_ids: List[int] = None):
        """Run complete ETL pipeline"""
        try:
            # Extract
            raw_data = self.extract_data(source_ids)
            
            # Transform
            processed_data = self.transform_data(raw_data)
            
            # Load
            self.load_data(processed_data)
            
            logger.info("ETL pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise
    
    def extract_data(self, source_ids: List[int] = None) -> List[Dict[str, Any]]:
        """Extract data from database"""
        return self.db_ops.get_raw_data(source_ids)
    
    def transform_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform raw data into processed format"""
        processed_data = []
        
        for data in raw_data:
            try:
                # Clean and preprocess text
                if data['extracted_text']:
                    cleaned_text = self.preprocessor.clean_text(data['extracted_text'])
                    features = self.feature_engineer.extract_features(cleaned_text)
                    
                    processed_data.append({
                        'raw_data_id': data['id'],
                        'processed_content': cleaned_text,
                        'features': features,
                        'data_type': self._determine_data_type(data),
                        'quality_score': self._calculate_quality_score(cleaned_text, features)
                    })
                
            except Exception as e:
                logger.warning(f"Failed to process data ID {data['id']}: {str(e)}")
                continue
        
        return processed_data
    
    def _determine_data_type(self, data: Dict[str, Any]) -> str:
        """Determine if data is structured or unstructured"""
        if data['file_type'] in ['csv', 'xlsx', 'xls']:
            return 'structured'
        return 'unstructured'
    
    def _calculate_quality_score(self, text: str, features: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        score = 0.0
        
        if text:
            score += 0.3  # Basic content exists
        
        if len(text) > 100:
            score += 0.3  # Sufficient length
        
        if features.get('word_count', 0) > 10:
            score += 0.2  # Meaningful content
        
        if features.get('readability_score', 0) > 0.3:
            score += 0.2  # Readable content
        
        return min(score, 1.0)
    
    def load_data(self, processed_data: List[Dict[str, Any]]):
        """Load processed data into database"""
        for data in processed_data:
            self.db_ops.insert_processed_data(data)