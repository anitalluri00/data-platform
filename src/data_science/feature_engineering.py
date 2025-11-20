import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from scipy import stats

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.vectorizers = {}
        self.feature_selectors = {}
        self.dimensionality_reducers = {}
    
    def extract_text_features(self, text_series: pd.Series, 
                            method: str = 'tfidf',
                            max_features: int = 1000) -> pd.DataFrame:
        """Extract features from text data"""
        try:
            if method == 'tfidf':
                if 'tfidf' not in self.vectorizers:
                    self.vectorizers['tfidf'] = TfidfVectorizer(
                        max_features=max_features,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                features = self.vectorizers['tfidf'].fit_transform(text_series.fillna(''))
                feature_names = self.vectorizers['tfidf'].get_feature_names_out()
            
            elif method == 'count':
                if 'count' not in self.vectorizers:
                    self.vectorizers['count'] = CountVectorizer(
                        max_features=max_features,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                features = self.vectorizers['count'].fit_transform(text_series.fillna(''))
                feature_names = self.vectorizers['count'].get_feature_names_out()
            
            else:
                raise ValueError(f"Unknown text feature extraction method: {method}")
            
            # Convert to DataFrame
            features_df = pd.DataFrame(
                features.toarray(),
                columns=[f"text_{method}_{name}" for name in feature_names],
                index=text_series.index
            )
            
            logger.info(f"Extracted {len(feature_names)} text features using {method}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting text features: {str(e)}")
            return pd.DataFrame()
    
    def create_interaction_features(self, df: pd.DataFrame,
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between numerical columns"""
        df_interactions = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df_interactions.columns and col2 in df_interactions.columns:
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df_interactions[col1] * df_interactions[col2]
                
                # Ratio interaction (avoid division by zero)
                ratio_name = f"{col1}_div_{col2}"
                df_interactions[ratio_name] = np.where(
                    df_interactions[col2] != 0,
                    df_interactions[col1] / df_interactions[col2],
                    0
                )
        
        logger.info(f"Created {len(feature_pairs) * 2} interaction features")
        return df_interactions
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                 columns: List[str],
                                 degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numerical columns"""
        df_poly = df.copy()
        
        for col in columns:
            if col in df_poly.columns:
                for deg in range(2, degree + 1):
                    poly_name = f"{col}_pow_{deg}"
                    df_poly[poly_name] = df_poly[col] ** deg
        
        logger.info(f"Created polynomial features for {len(columns)} columns up to degree {degree}")
        return df_poly
    
    def create_binning_features(self, df: pd.DataFrame,
                              columns: List[str],
                              bins: int = 5,
                              strategy: str = 'quantile') -> pd.DataFrame:
        """Create binning features for numerical columns"""
        df_binned = df.copy()
        
        for col in columns:
            if col in df_binned.columns:
                if strategy == 'quantile':
                    binned_name = f"{col}_binned_quantile"
                    df_binned[binned_name] = pd.qcut(
                        df_binned[col], bins, labels=False, duplicates='drop'
                    )
                elif strategy == 'uniform':
                    binned_name = f"{col}_binned_uniform"
                    df_binned[binned_name] = pd.cut(
                        df_binned[col], bins, labels=False
                    )
                elif strategy == 'kmeans':
                    from sklearn.cluster import KMeans
                    binned_name = f"{col}_binned_kmeans"
                    kmeans = KMeans(n_clusters=bins, random_state=42)
                    df_binned[binned_name] = kmeans.fit_predict(
                        df_binned[col].values.reshape(-1, 1)
                    )
        
        logger.info(f"Created binning features for {len(columns)} columns using {strategy} strategy")
        return df_binned
    
    def extract_date_features(self, df: pd.DataFrame,
                            date_columns: List[str]) -> pd.DataFrame:
        """Extract comprehensive features from date columns"""
        df_dates = df.copy()
        
        for col in date_columns:
            if col not in df_dates.columns:
                continue
            
            # Ensure datetime type
            df_dates[col] = pd.to_datetime(df_dates[col], errors='coerce')
            
            # Basic date components
            df_dates[f'{col}_year'] = df_dates[col].dt.year
            df_dates[f'{col}_month'] = df_dates[col].dt.month
            df_dates[f'{col}_day'] = df_dates[col].dt.day
            df_dates[f'{col}_dayofweek'] = df_dates[col].dt.dayofweek
            df_dates[f'{col}_dayofyear'] = df_dates[col].dt.dayofyear
            df_dates[f'{col}_week'] = df_dates[col].dt.isocalendar().week
            df_dates[f'{col}_quarter'] = df_dates[col].dt.quarter
            df_dates[f'{col}_is_weekend'] = (df_dates[col].dt.dayofweek >= 5).astype(int)
            df_dates[f'{col}_is_month_start'] = df_dates[col].dt.is_month_start.astype(int)
            df_dates[f'{col}_is_month_end'] = df_dates[col].dt.is_month_end.astype(int)
            
            # Time-based features (if time component exists)
            if df_dates[col].dt.time.nunique() > 1:
                df_dates[f'{col}_hour'] = df_dates[col].dt.hour
                df_dates[f'{col}_minute'] = df_dates[col].dt.minute
                df_dates[f'{col}_second'] = df_dates[col].dt.second
                
                # Time of day categories
                df_dates[f'{col}_time_of_day'] = pd.cut(
                    df_dates[col].dt.hour,
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                    include_lowest=True
                )
        
        logger.info(f"Extracted date features from {len(date_columns)} columns")
        return df_dates
    
    def create_aggregation_features(self, df: pd.DataFrame,
                                  groupby_columns: List[str],
                                  agg_columns: List[str],
                                  agg_functions: List[str] = None) -> pd.DataFrame:
        """Create aggregation features for grouped data"""
        if agg_functions is None:
            agg_functions = ['mean', 'std', 'min', 'max', 'count']
        
        df_agg = df.copy()
        
        for group_col in groupby_columns:
            if group_col not in df_agg.columns:
                continue
            
            for agg_col in agg_columns:
                if agg_col not in df_agg.columns:
                    continue
                
                # Group by and calculate aggregations
                group_agg = df_agg.groupby(group_col)[agg_col].agg(agg_functions)
                group_agg.columns = [f"{agg_col}_{func}_by_{group_col}" for func in agg_functions]
                
                # Merge back to original dataframe
                df_agg = df_agg.merge(
                    group_agg, 
                    how='left', 
                    left_on=group_col, 
                    right_index=True
                )
        
        logger.info(f"Created aggregation features for {len(agg_columns)} columns grouped by {len(groupby_columns)}")
        return df_agg
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'mutual_info',
                       k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using various methods"""
        try:
            if method == 'mutual_info':
                if 'mutual_info' not in self.feature_selectors:
                    self.feature_selectors['mutual_info'] = SelectKBest(
                        score_func=mutual_info_classif, k=min(k, X.shape[1])
                    )
                X_selected = self.feature_selectors['mutual_info'].fit_transform(X, y)
                selected_mask = self.feature_selectors['mutual_info'].get_support()
            
            elif method == 'f_classif':
                if 'f_classif' not in self.feature_selectors:
                    self.feature_selectors['f_classif'] = SelectKBest(
                        score_func=f_classif, k=min(k, X.shape[1])
                    )
                X_selected = self.feature_selectors['f_classif'].fit_transform(X, y)
                selected_mask = self.feature_selectors['f_classif'].get_support()
            
            elif method == 'f_regression':
                if 'f_regression' not in self.feature_selectors:
                    self.feature_selectors['f_regression'] = SelectKBest(
                        score_func=f_regression, k=min(k, X.shape[1])
                    )
                X_selected = self.feature_selectors['f_regression'].fit_transform(X, y)
                selected_mask = self.feature_selectors['f_regression'].get_support()
            
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
            
            selected_features = X.columns[selected_mask].tolist()
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            logger.info(f"Selected {len(selected_features)} features using {method}")
            return X_selected_df, selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return X, X.columns.tolist()
    
    def reduce_dimensionality(self, X: pd.DataFrame,
                            n_components: int = 2,
                            method: str = 'pca') -> pd.DataFrame:
        """Reduce dimensionality of features"""
        try:
            if method == 'pca':
                if 'pca' not in self.dimensionality_reducers:
                    self.dimensionality_reducers['pca'] = PCA(
                        n_components=min(n_components, X.shape[1])
                    )
                X_reduced = self.dimensionality_reducers['pca'].fit_transform(X)
                columns = [f'pca_component_{i+1}' for i in range(X_reduced.shape[1])]
            
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
            X_reduced_df = pd.DataFrame(X_reduced, columns=columns, index=X.index)
            
            logger.info(f"Reduced dimensionality to {n_components} components using {method}")
            return X_reduced_df
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {str(e)}")
            return X
    
    def create_feature_engineering_pipeline(self, config: Dict[str, Any]) -> callable:
        """Create a feature engineering pipeline based on configuration"""
        def pipeline(df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
            df_engineered = df.copy()
            
            # Text feature extraction
            if config.get('extract_text_features'):
                text_columns = config.get('text_columns', [])
                for text_col in text_columns:
                    if text_col in df_engineered.columns:
                        text_features = self.extract_text_features(
                            df_engineered[text_col],
                            config.get('text_method', 'tfidf'),
                            config.get('max_text_features', 1000)
                        )
                        df_engineered = pd.concat([df_engineered, text_features], axis=1)
            
            # Interaction features
            if config.get('create_interactions'):
                interaction_pairs = config.get('interaction_pairs', [])
                df_engineered = self.create_interaction_features(df_engineered, interaction_pairs)
            
            # Polynomial features
            if config.get('create_polynomials'):
                poly_columns = config.get('polynomial_columns', [])
                df_engineered = self.create_polynomial_features(
                    df_engineered, poly_columns, config.get('polynomial_degree', 2)
                )
            
            # Binning features
            if config.get('create_binning'):
                binning_columns = config.get('binning_columns', [])
                df_engineered = self.create_binning_features(
                    df_engineered, binning_columns, config.get('n_bins', 5)
                )
            
            # Feature selection (if target is provided)
            if config.get('select_features') and target is not None:
                feature_columns = [col for col in df_engineered.columns 
                                if col not in config.get('exclude_columns', [])]
                X_selected, selected_features = self.select_features(
                    df_engineered[feature_columns],
                    target,
                    config.get('selection_method', 'mutual_info'),
                    config.get('n_features', 10)
                )
                # Keep only selected features
                df_engineered = pd.concat([
                    df_engineered[config.get('exclude_columns', [])],
                    X_selected
                ], axis=1)
            
            return df_engineered
        
        return pipeline