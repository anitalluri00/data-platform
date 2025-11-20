from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class MLModelManager:
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
    
    def train_classification_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = "random_forest") -> Dict[str, Any]:
        """Train classification model"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_name == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            self.models[model_name] = model
            self.model_metrics[model_name] = metrics
            
            logger.info(f"Trained {model_name} with accuracy: {metrics['accuracy']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error training classification model: {str(e)}")
            raise
    
    def train_regression_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = "random_forest") -> Dict[str, Any]:
        """Train regression model"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_name == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred)
            }
            
            self.models[model_name] = model
            self.model_metrics[model_name] = metrics
            
            logger.info(f"Trained {model_name} with R2 score: {metrics['r2_score']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error training regression model: {str(e)}")
            raise
    
    def train_clustering_model(self, X: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """Train clustering model"""
        try:
            model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = model.fit_predict(X)
            
            metrics = {
                'inertia': model.inertia_,
                'n_clusters': n_clusters,
                'cluster_sizes': np.bincount(clusters)
            }
            
            self.models['kmeans'] = model
            self.model_metrics['kmeans'] = metrics
            
            logger.info(f"Trained KMeans with {n_clusters} clusters")
            
            return {
                'model': model,
                'clusters': clusters,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error training clustering model: {str(e)}")
            raise
    
    def save_model(self, model_name: str, file_path: str):
        """Save trained model to file"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], file_path)
            logger.info(f"Saved model {model_name} to {file_path}")
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def load_model(self, model_name: str, file_path: str):
        """Load model from file"""
        self.models[model_name] = joblib.load(file_path)
        logger.info(f"Loaded model {model_name} from {file_path}")