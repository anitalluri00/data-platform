import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from database.models import Base, SessionLocal
from utils.helpers import DataHelpers

logger = logging.getLogger(__name__)

class DataStorage:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.outputs_path = self.base_path / "outputs"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.outputs_path.mkdir(parents=True, exist_ok=True)
    
    def store_raw_file(self, file_path: str, category: str = "general") -> str:
        """Store raw file in organized directory structure"""
        try:
            file_name = Path(file_path).name
            category_path = self.raw_path / category
            category_path.mkdir(exist_ok=True)
            
            destination = category_path / file_name
            
            # Copy file to storage
            import shutil
            shutil.copy2(file_path, destination)
            
            logger.info(f"Stored raw file: {destination}")
            return str(destination)
            
        except Exception as e:
            logger.error(f"Error storing raw file {file_path}: {str(e)}")
            raise
    
    def store_processed_data(self, data: Any, filename: str, 
                           format_type: str = "parquet") -> str:
        """Store processed data in various formats"""
        try:
            file_path = self.processed_path / f"{filename}.{format_type}"
            
            if isinstance(data, pd.DataFrame):
                if format_type == "parquet":
                    data.to_parquet(file_path, index=False)
                elif format_type == "csv":
                    data.to_csv(file_path, index=False)
                elif format_type == "json":
                    data.to_json(file_path, orient='records', indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            else:
                # Store as JSON for non-DataFrame data
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Stored processed data: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing processed data: {str(e)}")
            raise
    
    def store_analysis_output(self, result: Dict[str, Any], 
                            report_name: str) -> str:
        """Store analysis outputs and reports"""
        try:
            # Store JSON report
            json_path = self.outputs_path / f"{report_name}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Store summary as text
            summary_path = self.outputs_path / f"{report_name}_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(self._generate_report_summary(result))
            
            logger.info(f"Stored analysis output: {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"Error storing analysis output: {str(e)}")
            raise
    
    def _generate_report_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable report summary"""
        summary = []
        summary.append("DATA ANALYSIS REPORT SUMMARY")
        summary.append("=" * 50)
        
        if 'metrics' in result:
            summary.append("\nMETRICS:")
            for key, value in result['metrics'].items():
                summary.append(f"  {key}: {value}")
        
        if 'insights' in result:
            summary.append("\nKEY INSIGHTS:")
            for insight in result['insights'][:5]:  # Top 5 insights
                summary.append(f"  • {insight}")
        
        if 'recommendations' in result:
            summary.append("\nRECOMMENDATIONS:")
            for rec in result['recommendations'][:3]:  # Top 3 recommendations
                summary.append(f"  • {rec}")
        
        summary.append(f"\nGenerated at: {pd.Timestamp.now()}")
        return "\n".join(summary)
    
    def cleanup_old_files(self, days_old: int = 30):
        """Clean up files older than specified days"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            
            for directory in [self.raw_path, self.processed_path, self.outputs_path]:
                for file_path in directory.rglob('*'):
                    if file_path.is_file():
                        file_time = file_path.stat().st_mtime
                        if file_time < cutoff_time:
                            file_path.unlink()
                            logger.info(f"Cleaned up old file: {file_path}")
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

class DatabaseStorage:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.session = SessionLocal()
    
    def store_dataframe(self, df: pd.DataFrame, table_name: str, 
                       if_exists: str = 'replace') -> bool:
        """Store DataFrame in database table"""
        try:
            df.to_sql(
                table_name, 
                self.engine, 
                if_exists=if_exists, 
                index=False,
                method='multi'
            )
            logger.info(f"Stored DataFrame in table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing DataFrame in {table_name}: {str(e)}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def backup_table(self, table_name: str, backup_suffix: str = None) -> bool:
        """Create backup of database table"""
        try:
            if backup_suffix is None:
                backup_suffix = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            backup_table = f"{table_name}_backup_{backup_suffix}"
            
            # Create backup using SQL
            with self.engine.connect() as connection:
                connection.execute(text(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}"))
                connection.commit()
            
            logger.info(f"Created backup: {backup_table}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up table {table_name}: {str(e)}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about database table"""
        try:
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            row_count = self.execute_query(count_query)[0]['row_count']
            
            # Get column information
            column_query = f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """
            columns = self.execute_query(column_query)
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'columns': columns,
                'size_mb': self._get_table_size(table_name)
            }
            
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            return {}
    
    def _get_table_size(self, table_name: str) -> float:
        """Get approximate table size in MB"""
        try:
            # MySQL specific query for table size
            size_query = f"""
                SELECT 
                    ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS size_mb
                FROM information_schema.TABLES 
                WHERE TABLE_NAME = '{table_name}'
            """
            result = self.execute_query(size_query)
            return result[0]['size_mb'] if result else 0.0
            
        except:
            return 0.0

class DistributedStorage:
    """Mock class for distributed storage systems"""
    
    def __init__(self):
        self.partitions = {}
    
    def store_distributed(self, data: Any, key: str, 
                         partition_key: str = None) -> bool:
        """Store data in distributed manner"""
        try:
            if partition_key is None:
                partition_key = "default"
            
            if partition_key not in self.partitions:
                self.partitions[partition_key] = {}
            
            self.partitions[partition_key][key] = data
            logger.info(f"Stored data in partition {partition_key} with key {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error in distributed storage: {str(e)}")
            return False
    
    def get_distributed(self, key: str, partition_key: str = None) -> Any:
        """Retrieve data from distributed storage"""
        try:
            if partition_key is None:
                partition_key = "default"
            
            return self.partitions.get(partition_key, {}).get(key)
            
        except Exception as e:
            logger.error(f"Error retrieving from distributed storage: {str(e)}")
            return None