FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    default-mysql-client \
    libmariadb-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/outputs reports logs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Install MySQL client for health checks
RUN pip install mysqlclient

# Run the application with MySQL wait
CMD ["sh", "-c", "python wait-for-mysql.py && streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0"]
