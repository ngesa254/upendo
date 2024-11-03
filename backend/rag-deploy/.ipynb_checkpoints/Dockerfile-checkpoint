FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and constraints files
COPY requirements/ requirements/
COPY constraints.txt .

# Install dependencies with constraints
RUN pip install --no-cache-dir -r requirements/prod.txt -c constraints.txt

# Copy the application code
COPY src/ src/
COPY data/ data/
COPY main.py .

# Set environment variables
ENV PORT=8080
ENV ENVIRONMENT=production

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}