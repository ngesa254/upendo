



# Navigate to your rag-deploy directory
cd ~/upendo/backend/rag-deploy

# Enable necessary Google Cloud APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    aiplatform.googleapis.com

# Create service account for the application
gcloud iam service-accounts create rag-service-account \
    --display-name="RAG System Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:rag-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Deploy using Cloud Build
gcloud builds submit --config deployment/prod/cloudbuild.yaml





cd ~/upendo/backend/rag-deploy



upendo/backend/rag-deploy/
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── document_chat.py
├── tests/
│   ├── __init__.py
│   └── test_document_chat.py
├── data/
│   └── Africa_Developer_Ecosystem_Report_2021.pdf
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── deployment/
│   ├── dev/
│   │   └── cloudbuild.yaml
│   └── prod/
│       └── cloudbuild.yaml
├── .env.dev
├── .env.prod
├── Dockerfile
└── main.py

# Create constraints.txt in the root directory
touch constraints.txt


upendo/backend/rag-deploy/
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── document_chat.py
├── tests/
│   ├── __init__.py
│   └── test_document_chat.py
├── data/
│   └── Africa_Developer_Ecosystem_Report_2021.pdf
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── deployment/
│   ├── dev/
│   │   └── cloudbuild.yaml
│   └── prod/
│       └── cloudbuild.yaml
├── constraints.txt   # New file
├── .env.dev
├── .env.prod
├── Dockerfile
└── main.py

# Create service account
gcloud iam service-accounts create rag-service-account \
    --display-name="RAG System Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:rag-service-account@angelic-bee-193823.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:rag-service-account@angelic-bee-193823.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
    

gcloud auth login

cd ~/upendo/backend/rag-deploy

# Build and deploy
gcloud builds submit --tag gcr.io/angelic-bee-193823/rag-system

gcloud run deploy rag-system \
    --image gcr.io/angelic-bee-193823/rag-system \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --timeout 3600 \
    --allow-unauthenticated
    
    
    


First, let's set the service account as active:

bashCopy# Set the service account as active
gcloud config set account 534297186371-compute@developer.gserviceaccount.com

Let's grant the necessary permissions:

bashCopy# Grant Cloud Build permissions
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:534297186371-compute@developer.gserviceaccount.com" \
    --role="roles/cloudbuild.builds.builder"

# Grant Storage permissions
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:534297186371-compute@developer.gserviceaccount.com" \
    --role="roles/storage.admin"

# Grant Cloud Run permissions
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:534297186371-compute@developer.gserviceaccount.com" \
    --role="roles/run.admin"

Create a Cloud Storage bucket for build logs:

bashCopy# Create a bucket for Cloud Build logs
gsutil mb -p angelic-bee-193823 gs://angelic-bee-193823_cloudbuild

Now try building and deploying again:

bashCopy# Switch back to your user account
gcloud config set account marvinngesa@gmail.com

# Build and deploy
gcloud builds submit --tag gcr.io/angelic-bee-193823/rag-system

gcloud run deploy rag-system \
    --image gcr.io/angelic-bee-193823/rag-system \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --timeout 3600 \
    --allow-unauthenticated
If you still encounter issues, you can try enabling the required APIs directly:
bashCopy# Enable required APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    aiplatform.googleapis.com \
    storage-api.googleapis.com \
    storage-component.googleapis.com
Alternatively, you can use the manual deployment method:
bashCopy# Build the container locally
docker build -t rag-system .

# Tag the image
docker tag rag-system gcr.io/angelic-bee-193823/rag-system

# Configure docker to use gcloud credentials
gcloud auth configure-docker

# Push to Container Registry
docker push gcr.io/angelic-bee-193823/rag-system

# Deploy to Cloud Run
gcloud run deploy rag-system \
    --image gcr.io/angelic-bee-193823/rag-system \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --timeout 3600 \
    --allow-unauthenticated
    
    
    
# Grant yourself Cloud Run Admin role
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="user:marvinngesa@gmail.com" \
    --role="roles/run.admin"

# Grant yourself Service Account User role
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="user:marvinngesa@gmail.com" \
    --role="roles/iam.serviceAccountUser"



I see the issue. You need to use your personal account (marvinngesa@gmail.com) to grant permissions since the service account doesn't have IAM admin rights. Let's fix this:

First, switch back to your personal account:

bashCopy# Switch to your personal account
gcloud config set account marvinngesa@gmail.com

Now grant the permissions using your personal account:

bashCopy# Grant Cloud Build permissions
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:534297186371-compute@developer.gserviceaccount.com" \
    --role="roles/cloudbuild.builds.builder"

# Grant Storage permissions
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:534297186371-compute@developer.gserviceaccount.com" \
    --role="roles/storage.admin"

# Grant Cloud Run permissions
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="serviceAccount:534297186371-compute@developer.gserviceaccount.com" \
    --role="roles/run.admin"

Create the Cloud Storage bucket:

bashCopy# Create bucket for Cloud Build
gsutil mb -p angelic-bee-193823 gs://angelic-bee-193823_cloudbuild-logs

Try building again:

bashCopy# Build and deploy
gcloud builds submit --tag gcr.io/angelic-bee-193823/rag-system

gcloud run deploy rag-system \
    --image gcr.io/angelic-bee-193823/rag-system \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --timeout 3600 \
    --allow-unauthenticated



gcloud run deploy rag-system \
    --image gcr.io/angelic-bee-193823/rag-system \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --allow-unauthenticated \
    --set-env-vars="ENVIRONMENT=production"
    
    
    
    
    
    # Grant Cloud Run Admin role
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="user:marvinngesa@gmail.com" \
    --role="roles/run.admin" \
    --condition=None

# Grant Service Account User role
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="user:marvinngesa@gmail.com" \
    --role="roles/iam.serviceAccountUser" \
    --condition=None

# Grant Cloud Run Invoker role
gcloud projects add-iam-policy-binding angelic-bee-193823 \
    --member="user:marvinngesa@gmail.com" \
    --role="roles/run.invoker" \
    --condition=None
    
    
gcloud services enable run.googleapis.com


gcloud run deploy rag-system \
    --image gcr.io/angelic-bee-193823/rag-system \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --service-account 534297186371-compute@developer.gserviceaccount.com \
    --set-env-vars="ENVIRONMENT=production" \
    --allow-unauthenticated