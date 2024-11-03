# FROM python:3.10-slim as builder

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy requirements and constraints files
# COPY requirements/ requirements/
# COPY constraints.txt .

# # Install dependencies with constraints
# RUN pip install --no-cache-dir -r requirements/prod.txt -c constraints.txt

# # Copy the application code
# COPY src/ src/
# COPY data/ data/
# COPY main.py .

# # Set environment variables
# ENV PORT=8080
# ENV ENVIRONMENT=production

# # Run the application
# CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}



# FROM python:3.10-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy requirements files
# COPY requirements/ requirements/
# COPY constraints.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements/prod.txt -c constraints.txt

# # Copy application code
# COPY src/ src/
# COPY data/ data/
# COPY main.py .

# # Set environment variables
# ENV PORT=8080
# ENV ENVIRONMENT=production

# # Run the application with proper timeout and workers
# CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 300


# FROM python:3.10-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy requirements files
# COPY requirements/ requirements/
# COPY constraints.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements/prod.txt -c constraints.txt

# # Copy application code
# COPY src/ src/
# COPY data/ data/
# COPY main.py .

# # Set environment variables
# ENV PORT=8080
# ENV ENVIRONMENT=production
# ENV PYTHONUNBUFFERED=1

# # Set health check timeout
# EXPOSE $PORT

# # Run with increased timeout
# CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120 --workers 1





# FROM python:3.10-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy requirements files
# COPY requirements/ requirements/
# COPY constraints.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements/prod.txt -c constraints.txt

# # Copy application code
# COPY src/ src/
# COPY data/ data/
# COPY main.py .

# # Set environment variables
# ENV PORT=8080
# ENV ENVIRONMENT=production
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# # Health check configuration
# ENV HEALTH_CHECK_PATH=/health
# ENV HEALTH_CHECK_INTERVAL=30s
# ENV HEALTH_CHECK_TIMEOUT=10s

# # Expose port
# EXPOSE $PORT

# # Run the application
# CMD exec gunicorn main:app \
#     --bind :$PORT \
#     --workers 1 \
#     --worker-class uvicorn.workers.UvicornWorker \
#     --timeout 0 \
#     --graceful-timeout 300 \
#     --keep-alive 5 \
#     --log-level info




FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}