**Titanic Survival Predictor**  
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

Project Structure  

├── app.py                     # Flask application  

├── TitanicSurvivalPredictionsModel.py  # ML model and pipeline  

├── templates/  

│   ├── base.html  

│   ├── index.html  

│   ├── about.html  

│   ├── help.html  

│   ├── status.html  

│   └── train.html  

├── static/  

│   └── style.css  

├── tests/  

│   └── test_model.py          # Unit tests  

└── README.md  



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
