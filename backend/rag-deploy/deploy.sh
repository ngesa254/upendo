#!/bin/bash
set -e  # Exit on error

# Configuration
PROJECT_ID="angelic-bee-193823"
REGION="us-central1"
SERVICE_NAME="rag-system"

echo "Starting deployment process..."
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Verify we're in the correct directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found! Make sure you're in the rag-deploy directory."
    exit 1
fi

# Build the image using Cloud Build
echo "Building container using Cloud Build..."
gcloud builds submit \
    --timeout=30m \
    --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 80 \
    --set-env-vars="ENVIRONMENT=production" \
    --allow-unauthenticated

# Get the service URL
echo "Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo "Service deployed at: ${SERVICE_URL}"

# Perform health check
echo "Performing health check..."
for i in {1..5}; do
    echo "Health check attempt $i..."
    if curl -s "${SERVICE_URL}/health" | grep -q "healthy"; then
        echo "Health check passed!"
        exit 0
    fi
    sleep 5
done

echo "Warning: Health check did not pass after 5 attempts"
exit 1
