import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str, 
                  remove_stopwords: bool = True,
                  remove_punctuation: bool = True,
                  lowercase: bool = True,
                  stem: bool = False,
                  lemmatize: bool = True) -> str:
        """Clean and preprocess text data"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        try:
            # Convert to lowercase
            if lowercase:
                text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Remove punctuation
            if remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Stemming or Lemmatization
            if stem:
                tokens = [self.stemmer.stem(token) for token in tokens]
            elif lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return ' '.join(tokens)
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            return text
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        if strategy is None:
            strategy = {}
        
        df_clean = df.copy()
        
        for column in df_clean.columns:
            col_strategy = strategy.get(column, 'auto')
            
            if col_strategy == 'auto':
                # Auto-detect best strategy based on data type
                if pd.api.types.is_numeric_dtype(df_clean[column]):
                    col_strategy = 'mean'
                else:
                    col_strategy = 'most_frequent'
            
            if col_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[column])
            else:
                if column not in self.imputers:
                    if col_strategy == 'mean':
                        self.imputers[column] = SimpleImputer(strategy='mean')
                    elif col_strategy == 'median':
                        self.imputers[column] = SimpleImputer(strategy='median')
                    elif col_strategy == 'most_frequent':
                        self.imputers[column] = SimpleImputer(strategy='most_frequent')
                    elif col_strategy == 'constant':
                        self.imputers[column] = SimpleImputer(strategy='constant', fill_value=0)
                    else:
                        continue
                
                # Reshape for single column
                column_data = df_clean[column].values.reshape(-1, 1)
                df_clean[column] = self.imputers[column].fit_transform(column_data).flatten()
        
        logger.info(f"Handled missing values for {len(df_clean.columns)} columns")
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                   columns: List[str] = None,
                                   method: str = 'label') -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        if columns is None:
            # Auto-detect categorical columns
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in columns:
            if column not in df_encoded.columns:
                continue
            
            if method == 'label':
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()
                df_encoded[column] = self.encoders[column].fit_transform(df_encoded[column].astype(str))
            
            elif method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[column], prefix=column)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[column])
            
            elif method == 'frequency':
                # Frequency encoding
                freq_encoding = df_encoded[column].value_counts().to_dict()
                df_encoded[column] = df_encoded[column].map(freq_encoding)
        
        logger.info(f"Encoded {len(columns)} categorical columns using {method} encoding")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame,
                               columns: List[str] = None,
                               method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        
        if columns is None:
            # Auto-detect numerical columns
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in df_scaled.columns:
                continue
            
            if method == 'standard':
                if column not in self.scalers:
                    self.scalers[column] = StandardScaler()
                df_scaled[column] = self.scalers[column].fit_transform(
                    df_scaled[column].values.reshape(-1, 1)
                ).flatten()
            
            elif method == 'minmax':
                if column not in self.scalers:
                    self.scalers[column] = MinMaxScaler()
                df_scaled[column] = self.scalers[column].fit_transform(
                    df_scaled[column].values.reshape(-1, 1)
                ).flatten()
            
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                if column not in self.scalers:
                    self.scalers[column] = RobustScaler()
                df_scaled[column] = self.scalers[column].fit_transform(
                    df_scaled[column].values.reshape(-1, 1)
                ).flatten()
        
        logger.info(f"Scaled {len(columns)} numerical columns using {method} scaling")
        return df_scaled
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numerical columns"""
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_mask = pd.Series([False] * len(df_clean))
        
        for column in columns:
            if column not in df_clean.columns:
                continue
            
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                column_outliers = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
                outliers_mask = outliers_mask | column_outliers
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_clean[column]))
                column_outliers = z_scores > threshold
                outliers_mask = outliers_mask | column_outliers
        
        df_clean = df_clean[~outliers_mask]
        logger.info(f"Removed {outliers_mask.sum()} outliers from dataset")
        
        return df_clean
    
    def extract_datetime_features(self, df: pd.DataFrame,
                                datetime_columns: List[str]) -> pd.DataFrame:
        """Extract features from datetime columns"""
        df_featured = df.copy()
        
        for col in datetime_columns:
            if col not in df_featured.columns:
                continue
            
            # Convert to datetime
            df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')
            
            # Extract features
            df_featured[f'{col}_year'] = df_featured[col].dt.year
            df_featured[f'{col}_month'] = df_featured[col].dt.month
            df_featured[f'{col}_day'] = df_featured[col].dt.day
            df_featured[f'{col}_hour'] = df_featured[col].dt.hour
            df_featured[f'{col}_dayofweek'] = df_featured[col].dt.dayofweek
            df_featured[f'{col}_quarter'] = df_featured[col].dt.quarter
            df_featured[f'{col}_is_weekend'] = df_featured[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        logger.info(f"Extracted datetime features from {len(datetime_columns)} columns")
        return df_featured
    
    def create_preprocessing_pipeline(self, config: Dict[str, Any]) -> callable:
        """Create a preprocessing pipeline based on configuration"""
        def pipeline(df: pd.DataFrame) -> pd.DataFrame:
            df_processed = df.copy()
            
            # Handle missing values
            if config.get('handle_missing'):
                df_processed = self.handle_missing_values(
                    df_processed, 
                    config.get('missing_strategy', {})
                )
            
            # Encode categorical variables
            if config.get('encode_categorical'):
                df_processed = self.encode_categorical_variables(
                    df_processed,
                    config.get('categorical_columns'),
                    config.get('encoding_method', 'label')
                )
            
            # Scale numerical features
            if config.get('scale_numerical'):
                df_processed = self.scale_numerical_features(
                    df_processed,
                    config.get('numerical_columns'),
                    config.get('scaling_method', 'standard')
                )
            
            # Remove outliers
            if config.get('remove_outliers'):
                df_processed = self.remove_outliers(
                    df_processed,
                    config.get('outlier_columns'),
                    config.get('outlier_method', 'iqr'),
                    config.get('outlier_threshold', 1.5)
                )
            
            # Extract datetime features
            if config.get('extract_datetime_features'):
                df_processed = self.extract_datetime_features(
                    df_processed,
                    config.get('datetime_columns', [])
                )
            
            return df_processed
        
        return pipeline