import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class SurvivalPrediction():
    '''Titanic survival predictor with bagging -> RF, LR'''
    def __init__(self):
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__preprocessor = None
        self.__pipeline = None
        self.__model_RF = None
        self.__model_LR = None
        self.__y_pred = None
        
        self.__numerical_features = None
        self.__categorical_features = None
        
        
    def data_explain_analysis(self) -> None:
        print(f"Value counts")
        print(self.__y.value_counts())
    
    def load_data(self) -> None:
        '''Loading data from sns.'''
        try:
            self.__data = sns.load_dataset('titanic')
            print(self.__data.head())
            
        except  Exception:
            print(f"Loading data failed !!!")
    
    def split_data(self) -> None:
        '''Select features/target and splits data 80-20 %'''
        if self.__data is None or self.__data.empty:
            raise RuntimeError("Data not loaded")
        
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
        target = 'survived'
        
        self.__X = self.__data[features]
        self.__y = self.__data[target]
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.2, random_state=42, stratify=self.__y)
    
    def feature_importances(self) -> None:
        '''Extract and plot feature importances from the best RF estimator'''
        try:
            self.__model_LR.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.__categorical_features)
            feature_importances = self.__model_RF.best_estimator_['classifier'].feature_importances_

            # Combine the numerical and one-hot encoded categorical feature names
            feature_names = self.__numerical_features + list(self.__model_LR.best_estimator_['preprocessor']
                                                    .named_transformers_['cat']
                                                    .named_steps['onehot']
                                                    .get_feature_names_out(self.__categorical_features))
    
            importance_df = pd.DataFrame({'Feature': feature_names,
                                        'Importance': feature_importances
                                        }).sort_values(by='Importance', ascending=False)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            plt.gca().invert_yaxis() 
            plt.title('Most Important Features in predicting whether a passenger survived')
            plt.xlabel('Importance Score')
            plt.show()
        except Exception:
            print("Error in feature_importance !!!")
            

    def preprocessing(self) -> None:
        '''
            Build ColumnTransformer using simple imputation + scaling / one-hot encoding
            Select dtypes from train set and preserve the order
        '''
        self.__numerical_features = self.__X_train.select_dtypes(include=['number']).columns.tolist()
        self.__categorical_features = self.__X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # preprocessing pipelines for both feature types
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.__preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.__numerical_features),
                ('cat', categorical_transformer, self.__categorical_features)
            ])
        
    def train_RF(self) -> None:
        '''Train RandomForest with GridSearchCV && '''
        try:           
            self.__pipeline = Pipeline(steps=[
                ('preprocessor', self.__preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        
            #prepruning
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }
        
            #Perform grid search cross-validation && fit to the best model
            cv = StratifiedKFold(n_splits=5, shuffle=True) # 5/10 folds
        
            self.__model_RF = GridSearchCV(estimator=self.__pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
        except Exception:
            print(f"Error during build the model !!!")
        
        self.__model_RF.fit(self.__X_train, self.__y_train)
    
    def train_logistic_regression(self) -> None:
        '''Train Logistic Regression using GridSearchCV over penalties and class weights'''
        try:
            self.__pipeline = Pipeline(steps=[
                ('preprocessor', self.__preprocessor),
                ('classifier', LogisticRegression(random_state=42, max_iter=2000))
            ])

            # Define a new grid with Logistic Regression parameters
            param_grid = {
                'classifier__solver' : ['liblinear'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__class_weight' : [None, 'balanced']
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            self.__model_LR = GridSearchCV(estimator=self.__pipeline,
                                        param_grid=param_grid,
                                        cv=cv,
                                        scoring='accuracy',
                                        verbose=2)
        except Exception:
            print(f"Error during build the Logistic Regression model !!!")
            
        self.__model_LR.fit(self.__X_train, self.__y_train)
        
    def LR_feature_coeff(self) -> None:
        coefficients = self.__model_LR.best_estimator_.named_steps['classifier'].coef_[0]

        # Combine numerical and categorical feature names
        numerical_feature_names = self.__numerical_features
        categorical_feature_names = (self.__model_LR.best_estimator_.named_steps['preprocessor']
                                            .named_transformers_['cat']
                                            .named_steps['onehot']
                                            .get_feature_names_out(self.__categorical_features)
                                    )
        feature_names = numerical_feature_names + list(categorical_feature_names)
        
        #ploting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
        plt.gca().invert_yaxis()
        plt.title('Feature Coefficient magnitudes for Logistic Regression model')
        plt.xlabel('Coefficient Magnitude')
        plt.show()

        
    def model_predict(self, model) -> None:
        if model == "RF":            
            self.__y_pred = self.__model_RF.predict(self.__X_test)
            print(classification_report(self.__y_test, self.__y_pred))
        elif model == "LR":
            self.__y_pred = self.__model_LR.predict(self.__X_test)
            print(classification_report(self.__y_test, self.__y_pred))
              
    def plot_confuxion_matrix(self) -> None:
        conf_matrix = confusion_matrix(self.__y_test, self.__y_pred)

        plt.figure()
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

        # Set the title and labels
        plt.title('Titanic Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def print_test_score(self, model) -> None:
        # Print test score
        if model == "RF":            
            test_score = self.__model_RF.score(self.__X_test, self.__y_test)
            print(f"\nTest set accuracy of Random Forest: {test_score:.2%}")
        elif model == "LR":
            test_score = self.__model_LR.best_estimator_.score(self.__X_test, self.__y_test)
            print(f"\nTest set accuracy of Logistic Regression: {test_score:.2%}")
        
    def run_pipeline(self):
        self.load_data()
        self.split_data()
        self.data_explain_analysis()
        self.preprocessing()
        
        #Random Forest
        self.train_RF()
        self.model_predict("RF")
        self.plot_confuxion_matrix()
        self.print_test_score("RF")
        
        #feature importances
        self.feature_importances()
        
        #Logistic Regression
        self.train_logistic_regression()
        self.LR_feature_coeff()
        self.model_predict("LR")
        self.plot_confuxion_matrix()
        self.print_test_score("LR")
            
if __name__ == "__main__":
    model = SurvivalPrediction()
    model.run_pipeline()
    