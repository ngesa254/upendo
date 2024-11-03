#!/bin/bash
set -e  # Exit on error

# Configuration
PROJECT_ID="534297186371"
REGION="us-central1"
SERVICE_NAME="rag-system"
SERVICE_ACCOUNT="${PROJECT_ID}-compute@developer.gserviceaccount.com"

echo "Starting deployment process..."

# Verify we're in the correct directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found! Make sure you're in the rag-deploy directory."
    exit 1
fi

# Verify requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Build and push the container
echo "Building and pushing container..."
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --service-account ${SERVICE_ACCOUNT} \
    --set-env-vars="ENVIRONMENT=production" \
    --allow-unauthenticated \
    --min-instances=1 \
    --max-instances=10

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo "Service deployed at: ${SERVICE_URL}"

# Basic health check
echo "Performing health check..."
curl -s "${SERVICE_URL}/health"

echo -e "\nDeployment completed! Test your endpoints at:"
echo "Health check: ${SERVICE_URL}/health"
echo "Chat endpoint: ${SERVICE_URL}/chat"
