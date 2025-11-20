#!/usr/bin/env python3
import os
import sys
import time
import MySQLdb
from config.settings import settings

def wait_for_mysql():
    """Wait for MySQL to be ready"""
    max_retries = 30
    retry_interval = 5
    
    for i in range(max_retries):
        try:
            conn = MySQLdb.connect(
                host=settings.MYSQL_HOST,
                port=int(settings.MYSQL_PORT),
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE
            )
            conn.close()
            print("✅ MySQL is ready!")
            return True
        except Exception as e:
            print(f"⏳ Waiting for MySQL... (attempt {i+1}/{max_retries}) - {str(e)}")
            time.sleep(retry_interval)
    
    print("❌ MySQL failed to start within the expected time")
    return False

if __name__ == "__main__":
    if wait_for_mysql():
        sys.exit(0)
    else:
        sys.exit(1)
