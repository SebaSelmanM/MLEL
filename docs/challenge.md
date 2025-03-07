# Challenge Documentation

## Part I: Bug Fixes and Model Selection

**Objective: Transcribe the exploration.ipynb notebook into model.py, fix any bugs, and choose the best model.**

Bug Found
In the function get_rate_from_column, the code used tot / delays[name] instead of delays[name] / tot.
This resulted in an inverted delay rate, leading to incorrect reports.
Solution
Swap the division to rates[name] = (delays[name] / tot).

Model Selection
We tested XGBoost and Logistic Regression.
We picked XGBoost with class balancing (scale_pos_weight) and the top 10 most important features based on model analysis.
This improved recall for the minority class (delay=1).
Best Practices
Separated preprocessing into individual functions (get_period_day, is_high_season, etc.).
Used clear docstrings and return types in model.py.
Handled exceptions (e.g., predicting without training the model).

## Part II: FastAPI Implementation

**Objective: Operationalize the model via an API using FastAPI.**

api.py File
Created a /health endpoint to check API status: returns {"status": "OK"}.
Implemented a /predict endpoint with Pydantic models (FlightItem, FlightsBatch) receiving flight data and returning {"predict": [...]}.
Added a RequestValidationError handler to return 400 for unknown columns or out-of-range values.
Endpoint Details
/predict: Accepts a batch of flights, calls model.preprocess(...) and model.predict(...), and responds with the key "predict".
Pydantic classes have extra = "forbid", ensuring any unknown columns lead to a validation error.
Testing (make api-test)
Adjusted the JSON structure to match what the test suite expects: {"flights": [...]}.
Returns a 400 error when encountering unknown or invalid fields; otherwise 200 with {"predict": [...]}.

## Part III: Deploying to GCP (Cloud Run)

**Objective: Deploy the API to Google Cloud Platform.**

Containerization (Docker)
Created a Dockerfile:
Installs dependencies via pip install -r requirements.txt.
Copies the code into the container (COPY . /app).
Uses uvicorn challenge.api:app --host 0.0.0.0 --port 8080 to serve the API.
Publishing to Container Registry
docker build -t gcr.io...
docker push gcr.io/...
Deploying on Cloud Run
gcloud run deploy ...
Obtained a URL 
Stress Testing (make stress-test)
Updated line 26 in the Makefile to point API_URL to the new Cloud Run URL.
Installed Locust for load testing.
make stress-test then sends concurrent requests against the deployed API.

## Part IV: CI/CD Implementation (GitHub Actions)

**Objective: Provide a proper CI/CD approach in the .github/workflows folder.**

Folder .github/workflows
ci.yml
Runs tests (make model-test, make api-test) on each push or pull request.
Uses actions/checkout@v3, actions/setup-python@v4, etc.
cd.yml
Triggered on pushes to main or via manual dispatch.
Authenticates with GCP (Service Account Key).
Builds and pushes the Docker image to gcr.io/<PROJECT_ID>.
Deploys to Cloud Run using gcloud run deploy.
Optionally, runs make stress-test to verify performance in production.
Benefits
CI ensures that code changes pass unit and integration tests before merging.
CD automates builds and deployments to Cloud Run after successful CI checks.
The pipeline also runs stress tests for reliability checks in production-like environments.
## Final Notes

Dependencies: Listed in requirements.txt.
Environment Variables: The Makefile references the API_URL on a given line for load tests.
Class Balancing: Adjust scale_pos_weight in XGBoost if the dataset becomes more or less imbalanced.
Cold Start: Cloud Run may experience a slower first request if idle (cold start).
Summary

Model: XGBoost with class balancing and top 10 features for better recall on the minority class.
API: FastAPI with Pydantic-based validation, responding with {"predict": [...]}.
Deployment: Docker + Cloud Run for a serverless approach.
CI/CD: GitHub Actions for automated testing and deployment.