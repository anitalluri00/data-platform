import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Callable
import logging
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import json

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.validation_rules = {}
        self.expectation_suites = {}
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          validation_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive DataFrame validation"""
        results = {
            'is_valid': True,
            'validation_errors': [],
            'warnings': [],
            'metrics': {},
            'summary': {}
        }
        
        try:
            # Basic structure validation
            structure_checks = self._validate_structure(df)
            results['summary'].update(structure_checks)
            
            if not structure_checks['has_data']:
                results['is_valid'] = False
                results['validation_errors'].append("DataFrame is empty")
                return results
            
            # Data quality validation
            quality_checks = self._validate_data_quality(df)
            results['summary'].update(quality_checks)
            
            # Custom validation rules
            if validation_rules:
                custom_checks = self._validate_custom_rules(df, validation_rules)
                results['summary'].update(custom_checks)
            
            # Calculate overall score
            results['quality_score'] = self._calculate_quality_score(results['summary'])
            
            logger.info(f"Data validation completed. Score: {results['quality_score']:.2f}")
            
        except Exception as e:
            results['is_valid'] = False
            results['validation_errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Data validation failed: {str(e)}")
        
        return results
    
    def _validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame structure"""
        return {
            'has_data': len(df) > 0,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics"""
        total_cells = df.size
        
        # Null analysis
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        
        # Data type consistency
        type_consistency = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                type_counts = df[col].apply(type).value_counts()
                type_consistency[col] = {
                    'is_consistent': len(type_counts) == 1,
                    'type_distribution': type_counts.to_dict()
                }
        
        return {
            'null_percentage': (total_nulls / total_cells) * 100 if total_cells > 0 else 0,
            'duplicate_percentage': (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0,
            'completeness_score': 100 - ((total_nulls / total_cells) * 100) if total_cells > 0 else 0,
            'type_consistency': type_consistency,
            'column_null_rates': (null_counts / len(df) * 100).to_dict()
        }
    
    def _validate_custom_rules(self, df: pd.DataFrame, 
                             rules: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom validation rules"""
        results = {}
        
        for rule_name, rule_config in rules.items():
            try:
                if rule_config['type'] == 'range_check':
                    result = self._validate_range(df, rule_config)
                elif rule_config['type'] == 'pattern_check':
                    result = self._validate_pattern(df, rule_config)
                elif rule_config['type'] == 'value_check':
                    result = self._validate_values(df, rule_config)
                else:
                    result = {'valid': False, 'error': f"Unknown rule type: {rule_config['type']}"}
                
                results[rule_name] = result
                
            except Exception as e:
                results[rule_name] = {'valid': False, 'error': str(e)}
        
        return results
    
    def _validate_range(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numerical ranges"""
        column = rule['column']
        min_val = rule.get('min')
        max_val = rule.get('max')
        
        if column not in df.columns:
            return {'valid': False, 'error': f"Column {column} not found"}
        
        violations = 0
        if min_val is not None:
            violations += (df[column] < min_val).sum()
        if max_val is not None:
            violations += (df[column] > max_val).sum()
        
        return {
            'valid': violations == 0,
            'violation_count': violations,
            'violation_percentage': (violations / len(df)) * 100
        }
    
    def _validate_pattern(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate string patterns using regex"""
        import re
        
        column = rule['column']
        pattern = rule['pattern']
        
        if column not in df.columns:
            return {'valid': False, 'error': f"Column {column} not found"}
        
        pattern_matches = df[column].astype(str).str.match(pattern, na=False)
        violations = (~pattern_matches).sum()
        
        return {
            'valid': violations == 0,
            'violation_count': violations,
            'violation_percentage': (violations / len(df)) * 100
        }
    
    def _validate_values(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate allowed values"""
        column = rule['column']
        allowed_values = rule.get('allowed_values', [])
        disallowed_values = rule.get('disallowed_values', [])
        
        if column not in df.columns:
            return {'valid': False, 'error': f"Column {column} not found"}
        
        violations = 0
        
        if allowed_values:
            violations += (~df[column].isin(allowed_values)).sum()
        
        if disallowed_values:
            violations += (df[column].isin(disallowed_values)).sum()
        
        return {
            'valid': violations == 0,
            'violation_count': violations,
            'violation_percentage': (violations / len(df)) * 100
        }
    
    def _calculate_quality_score(self, summary: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        score = 100.0
        
        # Penalize for nulls
        null_penalty = summary.get('null_percentage', 0)
        score -= null_penalty * 0.5
        
        # Penalize for duplicates
        duplicate_penalty = summary.get('duplicate_percentage', 0)
        score -= duplicate_penalty * 0.3
        
        # Penalize for type inconsistencies
        type_consistency = summary.get('type_consistency', {})
        inconsistent_columns = sum(1 for col_stats in type_consistency.values() 
                                if not col_stats.get('is_consistent', True))
        score -= inconsistent_columns * 5
        
        return max(0.0, min(100.0, score))

class SchemaValidator:
    def __init__(self):
        self.schemas = {}
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DataFrame against expected schema"""
        results = {
            'is_valid': True,
            'schema_errors': [],
            'schema_warnings': [],
            'matches': True
        }
        
        try:
            # Check column presence
            expected_columns = set(expected_schema.get('columns', {}).keys())
            actual_columns = set(df.columns)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            if missing_columns:
                results['is_valid'] = False
                results['matches'] = False
                results['schema_errors'].append(f"Missing columns: {missing_columns}")
            
            if extra_columns:
                results['schema_warnings'].append(f"Extra columns: {extra_columns}")
            
            # Check data types
            for col, expected_type in expected_schema.get('columns', {}).items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if not self._types_match(actual_type, expected_type):
                        results['schema_errors'].append(
                            f"Column {col}: expected {expected_type}, got {actual_type}"
                        )
                        results['is_valid'] = False
            
            # Check constraints
            constraints = expected_schema.get('constraints', {})
            for constraint_type, constraint_config in constraints.items():
                if constraint_type == 'not_null':
                    self._validate_not_null(df, constraint_config, results)
                elif constraint_type == 'unique':
                    self._validate_unique(df, constraint_config, results)
            
        except Exception as e:
            results['is_valid'] = False
            results['schema_errors'].append(f"Schema validation error: {str(e)}")
        
        return results
    
    def _types_match(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual type matches expected type"""
        type_mapping = {
            'int64': ['integer', 'int', 'bigint', 'int64'],
            'float64': ['float', 'double', 'numeric', 'float64'],
            'object': ['string', 'text', 'varchar', 'object'],
            'bool': ['boolean', 'bool'],
            'datetime64[ns]': ['datetime', 'timestamp', 'datetime64[ns]']
        }
        
        for compatible_types in type_mapping.values():
            if actual_type in compatible_types and expected_type in compatible_types:
                return True
        
        return False
    
    def _validate_not_null(self, df: pd.DataFrame, columns: List[str], results: Dict[str, Any]):
        """Validate NOT NULL constraints"""
        for column in columns:
            if column in df.columns:
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    results['schema_errors'].append(
                        f"NOT NULL violation in {column}: {null_count} null values"
                    )
                    results['is_valid'] = False
    
    def _validate_unique(self, df: pd.DataFrame, columns: List[str], results: Dict[str, Any]):
        """Validate UNIQUE constraints"""
        for column in columns:
            if column in df.columns:
                duplicate_count = df[column].duplicated().sum()
                if duplicate_count > 0:
                    results['schema_errors'].append(
                        f"UNIQUE violation in {column}: {duplicate_count} duplicates"
                    )
                    results['is_valid'] = False