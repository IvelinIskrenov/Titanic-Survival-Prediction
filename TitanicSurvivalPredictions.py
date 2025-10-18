import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


class TitanicSurvavalPrediction():
    
    def __init__(self):
        self.__data = None
        self.__X = None
        self.__y = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__preprocessor = None
        self.__model = None
        self.__y_pred = None
        
    def data_explain_analysis(self) -> None:
        print(f"Value counts")
        print(self.__y.value_counts())
    
    def load_data(self) -> None:
        try:
            self.__data = sns.load_dataset('titanic')
            print(self.__data.head())
            
        except  Exception:
            print(f"Loading data failed !!!")
    
    def split_data(self) -> None:
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
        target = 'survived'
        
        self.__X = self.__data[features]
        self.__y = self.__data[target]
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.2, random_state=42, stratify=self.__y)
    
    def preprocessing(self) -> None:
        
        numerical_features = self.__X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_features = self.__X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
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
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
    def train_RF(self) -> None:
        pipeline = Pipeline(steps=[
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
        
        self.__model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
        self.__model.fit(self.__X_train, self.__y_train)
        
    def model_predict(self) -> None:
        self.__y_pred = self.__model.predict(self.__X_test)
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

    def pipeline(self):
        self.load_data()
        self.split_data()
        #self.data_explain_analysis()
        self.preprocessing()
        self.train_RF()
        self.model_predict()
        self.plot_confuxion_matrix()
            
if __name__ == "__main__":
    model = TitanicSurvavalPrediction()
    model.pipeline()
    