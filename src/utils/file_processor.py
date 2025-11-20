import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import magic
import hashlib

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.mime = magic.Magic(mime=True)
    
    def detect_file_type(self, file_path: str) -> Dict[str, Any]:
        """Detect file type and metadata"""
        try:
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            mime_type = self.mime.from_file(file_path)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': file_size,
                'mime_type': mime_type,
                'file_hash': file_hash,
                'extension': Path(file_path).suffix.lower(),
                'created_time': file_stats.st_ctime,
                'modified_time': file_stats.st_mtime
            }
            
        except Exception as e:
            logger.error(f"Error detecting file type for {file_path}: {str(e)}")
            raise
    
    def _calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def validate_file(self, file_path: str, allowed_types: List[str] = None) -> Tuple[bool, str]:
        """Validate file type and integrity"""
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            file_info = self.detect_file_type(file_path)
            
            # Check file size
            if file_info['file_size'] == 0:
                return False, "File is empty"
            
            # Check allowed types
            if allowed_types and file_info['extension'] not in allowed_types:
                return False, f"File type {file_info['extension']} not allowed"
            
            return True, "File is valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def chunk_large_file(self, file_path: str, chunk_size_mb: int = 10) -> List[str]:
        """Split large files into smaller chunks"""
        chunk_size = chunk_size_mb * 1024 * 1024  # Convert to bytes
        chunk_files = []
        
        try:
            with open(file_path, 'rb') as original_file:
                chunk_index = 0
                while True:
                    chunk_data = original_file.read(chunk_size)
                    if not chunk_data:
                        break
                    
                    chunk_filename = f"{file_path}.part{chunk_index:04d}"
                    with open(chunk_filename, 'wb') as chunk_file:
                        chunk_file.write(chunk_data)
                    
                    chunk_files.append(chunk_filename)
                    chunk_index += 1
            
            logger.info(f"Split {file_path} into {len(chunk_files)} chunks")
            return chunk_files
            
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {str(e)}")
            raise
    
    def process_csv_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Process CSV file with various options"""
        try:
            # Common CSV reading parameters
            params = {
                'low_memory': False,
                'encoding': 'utf-8',
                'on_bad_lines': 'skip'
            }
            params.update(kwargs)
            
            df = pd.read_csv(file_path, **params)
            
            # Basic data cleaning
            df = self._clean_dataframe(df)
            
            logger.info(f"Processed CSV file {file_path} with {len(df)} rows")
            return df
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    df = self._clean_dataframe(df)
                    logger.info(f"Processed CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            raise
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            raise
    
    def process_excel_file(self, file_path: str, sheet_name: str = None) -> Dict[str, pd.DataFrame]:
        """Process Excel file and return dictionary of sheets"""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            
            for sheet in excel_file.sheet_names:
                if sheet_name is None or sheet == sheet_name:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    sheets[sheet] = self._clean_dataframe(df)
            
            logger.info(f"Processed Excel file {file_path} with {len(sheets)} sheets")
            return sheets
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic dataframe cleaning"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip()
        
        # Convert date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
        
        return df
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive file metadata"""
        file_info = self.detect_file_type(file_path)
        
        try:
            # Additional metadata based on file type
            if file_info['extension'] in ['.csv', '.xlsx', '.xls']:
                if file_info['extension'] == '.csv':
                    df = self.process_csv_file(file_path, nrows=5)  # Sample first 5 rows
                    file_info['columns'] = df.columns.tolist()
                    file_info['sample_data'] = df.to_dict('records')
                else:
                    sheets = self.process_excel_file(file_path)
                    file_info['sheets'] = list(sheets.keys())
                    if sheets:
                        first_sheet = next(iter(sheets.values()))
                        file_info['columns'] = first_sheet.columns.tolist()
                        file_info['sample_data'] = first_sheet.head().to_dict('records')
            
            elif file_info['extension'] in ['.txt', '.pdf', '.docx']:
                # For text files, get character count and encoding info
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    file_info['character_count'] = len(raw_data)
                    
                    # Try to detect encoding
                    try:
                        text_content = raw_data.decode('utf-8')
                        file_info['encoding'] = 'utf-8'
                    except UnicodeDecodeError:
                        try:
                            text_content = raw_data.decode('latin-1')
                            file_info['encoding'] = 'latin-1'
                        except UnicodeDecodeError:
                            file_info['encoding'] = 'unknown'
            
            return file_info
            
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata for {file_path}: {str(e)}")
            return file_info