**Titanic Survival Predictor**  

![CI/CD Status](https://github.com/iveliniskrenov/TitanicSurvivalPrediction/actions/workflows/CI_CD.yml/badge.svg)

A web-based application for predicting the survival of passengers on the Titanic using machine learning models. This project leverages Random Forest and Logistic Regression algorithms to predict survival based on passenger data.

**Features**

Interactive Web Interface: Built using Flask for easy navigation and interaction.

Model Training:            Train a Random Forest model directly from the web interface.

Evaluation & Status:       Check model training status and evaluate model performance.

Feature Analysis:          Visualize feature importance and coefficient magnitudes.

Help Section:              Detailed explanations of features used in the model.

Responsive Design:         User-friendly interface with clear navigation.

**Installation**
__Clone the repository:__

git clone <repository-url>
cd <repository-folder>

__Create a virtual environment (optional but recommended):__

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows


__Install dependencies:__

pip install -r requirements.txt

Required libraries include: Flask, scikit-learn, pandas, numpy, seaborn, matplotlib

**Usage**
__Start the Flask app:__

python app.py

__Navigate to the web interface:__

Open your browser and go to:
http://127.0.0.1:5000/

__Available Pages:__

Home: Overview of the Titanic Survival Predictor.

Train: Train the Random Forest model.

Status: Check if the model is trained.

Predict: Evaluate the model on the test set.

About: Learn more about the Titanic disaster.

Help: Detailed explanations of input features.

**Machine Learning Models**

Random Forest Classifier
Uses GridSearchCV for hyperparameter tuning.
Visualizes feature importance.
Supports cross-validation accuracy evaluation.

**Logistic Regression**

Uses GridSearchCV for hyperparameter tuning.
Visualizes feature coefficient magnitudes.
Supports cross-validation accuracy evaluation.

## New Update - 31.01.2026

## Automation & CI/CD

This repository is fully automated using **GitHub Actions** to ensure code quality and seamless deployment:

* **Continuous Integration (CI):**
    * **Linting:** Code is automatically checked for PEP8 compliance using `flake8`.
    * **Testing:** Automated unit tests are executed with `pytest` on every push to ensure the model logic remains intact.
* **Continuous Delivery (CD):**
    * **Dockerization:** Upon successful testing, the application is packaged into a **Docker Image**.
    * **Image Registry:** The image is automatically pushed to [Docker Hub](https://hub.docker.com/u/iveliniskrenov), making it ready for production.

## 🐳 How to Run via Docker

You don't need to install Python or any dependencies. Simply pull the pre-built image from Docker Hub and run it:

```bash
# Pull the latest image
docker pull iveliniskrenov/titanic-project:latest

# Run the container (Map port 5000)
docker run -p 5000:5000 iveliniskrenov/titanic-project
\```

Once running, the app will be available at `http://localhost:5000`.



**How It Works**

Data Loading: Loads Titanic dataset from seaborn.
Preprocessing: Handles missing values, scales numerical features, encodes categorical features.
Model Training: Trains both Random Forest and Logistic Regression with optimized hyperparameters.
Evaluation: Provides test accuracy, confusion matrix visualization, and feature importance plots.
Web Integration: Flask routes allow users to interact with the model and view results.

**Testing**  

Unit tests are implemented using unittest. To run tests:

python -m unittest discover tests

**Visualization**  

Feature Importance: Shows which features most influence Random Forest predictions.
Coefficient Magnitudes: Shows influence of features for Logistic Regression.
Confusion Matrix: Visualizes model performance on the test set.
