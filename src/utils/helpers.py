import json
import logging
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class DataHelpers:
    @staticmethod
    def generate_unique_id() -> str:
        """Generate a unique identifier"""
        return str(uuid.uuid4())
    
    @staticmethod
    def hash_data(data: str) -> str:
        """Generate SHA256 hash of data"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def safe_json_serialize(obj: Any) -> str:
        """Safely serialize object to JSON string"""
        def json_serializer(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(obj, default=json_serializer, ensure_ascii=False)
    
    @staticmethod
    def safe_json_deserialize(json_str: str) -> Any:
        """Safely deserialize JSON string"""
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON deserialization failed: {str(e)}")
            return {}
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    @staticmethod
    def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics for a DataFrame"""
        total_cells = df.size
        null_count = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': total_cells,
            'null_percentage': (null_count / total_cells) * 100 if total_cells > 0 else 0,
            'duplicate_percentage': (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_types': df.dtypes.astype(str).to_dict(),
            'column_stats': {}
        }
        
        # Column-level statistics
        for col in df.columns:
            col_stats = {
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'data_type': str(df[col].dtype)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
            elif pd.api.types.is_string_dtype(df[col]):
                col_stats['max_length'] = df[col].str.len().max()
            
            metrics['column_stats'][col] = col_stats
        
        return metrics
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Simple email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to remove invalid characters"""
        import re
        # Remove invalid characters for most filesystems
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        # Limit length
        return sanitized[:255]
    
    @staticmethod
    def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks of specified size"""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    @staticmethod
    def timeit(func):
        """Decorator to measure execution time"""
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                          exceptions: tuple = (Exception,)):
        """Decorator for retrying functions on exception"""
        import time
        from functools import wraps
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_retries - 1:
                            raise e
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                        time.sleep(delay)
                return None
            return wrapper
        return decorator

class DateTimeHelpers:
    @staticmethod
    def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
        """Parse date string with multiple format attempts"""
        if formats is None:
            formats = [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d-%m-%Y',
                '%m-%d-%Y',
                '%Y%m%d'
            ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    
    @staticmethod
    def get_time_ranges() -> Dict[str, tuple]:
        """Get common time ranges for analytics"""
        now = datetime.now()
        return {
            'last_24_hours': (now - timedelta(hours=24), now),
            'last_7_days': (now - timedelta(days=7), now),
            'last_30_days': (now - timedelta(days=30), now),
            'last_90_days': (now - timedelta(days=90), now),
            'current_month': (
                datetime(now.year, now.month, 1),
                now
            ),
            'previous_month': (
                datetime(now.year, now.month - 1, 1) if now.month > 1 else datetime(now.year - 1, 12, 1),
                datetime(now.year, now.month, 1) - timedelta(days=1)
            )
        }