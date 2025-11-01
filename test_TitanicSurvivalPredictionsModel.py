import unittest
from unittest.mock import patch, MagicMock
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


# If your file has another name, replace 'TitanicSurvivalPredictionsModel' with it
from TitanicSurvivalPredictionsModel import SurvivalPrediction


def small_data():
    return pd.DataFrame({
        'pclass': [1, 3, 2, 3, 1, 2],
        'sex': ['male', 'female', 'female', 'male', 'female', 'male'],
        'age': [22, 38, 26, 35, np.nan, 54],
        'sibsp': [1, 1, 0, 0, 0, 1],
        'parch': [0, 0, 0, 0, 2, 0],
        'fare': [7.25, 71.28, 7.92, 8.05, 53.10, 26.55],
        'class': ['Third', 'First', 'Second', 'Third', 'First', 'Second'],
        'who': ['man', 'woman', 'woman', 'man', 'woman', 'man'],
        'adult_male': [True, False, False, True, False, True],
        'alone': [False, False, True, True, False, True],
        'survived': [0, 1, 1, 0, 1, 0]
    })


class TestSurvivalPrediction(unittest.TestCase):

    def setUp(self):
        self.model = SurvivalPrediction()
        self.df = small_data()

    @patch('TitanicSurvivalPredictionsModel.sns.load_dataset')
    def test_load_data_assigns_dataframe(self, mock_load):
        # Mock sns.load_dataset to return our DataFrame
        mock_load.return_value = self.df.copy()
        self.model.load_data()
        #Check name-mangled __data is assigned
        self.assertIsNotNone(self.model._SurvivalPrediction__data)
        self.assertEqual(len(self.model._SurvivalPrediction__data), len(self.df))

    def test_split_data_raises_when_no_data(self):
        
        with self.assertRaises(RuntimeError):
            self.model.split_data()

    def test_split_data_populates_X_y_and_train_test(self):
        self.model._SurvivalPrediction__data = self.df.copy()
        self.model.split_data()

        X = self.model._SurvivalPrediction__X
        y = self.model._SurvivalPrediction__y

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        #-> train/test variables exist (name-mangled) ?
        self.assertTrue(hasattr(self.model, '_SurvivalPrediction__X_train'))
        self.assertTrue(hasattr(self.model, '_SurvivalPrediction__X_test'))
        self.assertTrue(hasattr(self.model, '_SurvivalPrediction__y_train'))
        self.assertTrue(hasattr(self.model, '_SurvivalPrediction__y_test'))

    def test_preprocessing_builds_column_transformer(self):
        #checking preprocessing() builds a ColumnTransformer based on X_train
        self.model._SurvivalPrediction__data = self.df.copy()
        self.model.split_data()
        self.model.preprocessing()

        num_feats = self.model._SurvivalPrediction__numerical_features
        cat_feats = self.model._SurvivalPrediction__categorical_features

        self.assertIsInstance(num_feats, list)
        self.assertIsInstance(cat_feats, list)
        self.assertGreater(len(num_feats), 0)
        self.assertGreater(len(cat_feats), 0)

        self.assertIsInstance(self.model._SurvivalPrediction__preprocessor, ColumnTransformer)
    # g 
    @patch('TitanicSurvivalPredictionsModel.GridSearchCV')
    def test_train_RF_sets_model_with_best_estimator(self, mock_gs_ctor):
        # Prepare data and preprocessor
        self.model._SurvivalPrediction__data = self.df.copy()
        self.model.split_data()
        self.model.preprocessing()

        # Create a MagicMock that mimics a GridSearchCV instance
        gs_instance = MagicMock()
        dummy_clf = MagicMock()
        dummy_clf.feature_importances_ = np.array([0.4, 0.3, 0.15, 0.1, 0.05])

        # Pipeline-like object: supports ['classifier'] and named_steps
        pipeline_like = MagicMock()
        pipeline_like.__getitem__.side_effect = lambda k: {'classifier': dummy_clf, 'preprocessor': MagicMock()}[k]
        pipeline_like.named_steps = {'classifier': dummy_clf, 'preprocessor': MagicMock()}

        gs_instance.best_estimator_ = pipeline_like
        gs_instance.fit.return_value = gs_instance
        mock_gs_ctor.return_value = gs_instance

        # Train (will use the mock GridSearchCV)
        self.model.train_RF()

        self.assertIsNotNone(self.model._SurvivalPrediction__model_RF)
        self.assertTrue(hasattr(self.model._SurvivalPrediction__model_RF, 'best_estimator_'))

    @patch('TitanicSurvivalPredictionsModel.GridSearchCV')
    def test_train_logistic_regression_sets_model_LR(self, mock_gs_ctor):
        # Prepare data and preprocessor
        self.model._SurvivalPrediction__data = self.df.copy()
        self.model.split_data()
        self.model.preprocessing()

        gs_instance = MagicMock()
        dummy_lr = MagicMock()
        dummy_lr.coef_ = np.array([[0.5, -0.2, 0.1, 0.0, -0.1]])

        pipeline_like = MagicMock()
        pipeline_like.named_steps = {'classifier': dummy_lr, 'preprocessor': MagicMock()}
        pipeline_like.__getitem__.side_effect = lambda k: pipeline_like.named_steps.get(k, MagicMock())

        gs_instance.best_estimator_ = pipeline_like
        gs_instance.fit.return_value = gs_instance
        mock_gs_ctor.return_value = gs_instance

        self.model.train_logistic_regression()

        self.assertIsNotNone(self.model._SurvivalPrediction__model_LR)
        self.assertTrue(hasattr(self.model._SurvivalPrediction__model_LR, 'best_estimator_'))

    def test_model_predict_works_for_RF_and_LR(self):
        # Prepare data and fake predictive models
        self.model._SurvivalPrediction__data = self.df.copy()
        self.model.split_data()

        fake_model = MagicMock()
        fake_model.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)

        self.model._SurvivalPrediction__model_RF = fake_model
        self.model._SurvivalPrediction__model_LR = fake_model

        # These calls should not raise exceptions
        self.model.model_predict('RF')
        self.model.model_predict('LR')

    def test_feature_importances_and_LR_coeff_no_exceptions(self):
        # Prepare data and preprocessing
        self.model._SurvivalPrediction__data = self.df.copy()
        self.model.split_data()
        self.model.preprocessing()

        dummy_clf = MagicMock()
        dummy_clf.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.06, 0.04])

        fake_onehot = MagicMock()
        fake_onehot.get_feature_names_out.return_value = np.array(['sex_male', 'class_First', 'who_man'])

        cat_transformer = MagicMock()
        cat_transformer.named_steps = {'onehot': fake_onehot}

        fake_preprocessor = MagicMock()
        fake_preprocessor.named_transformers_ = {'cat': cat_transformer}

        pipeline_like = MagicMock()
        pipeline_like.__getitem__.side_effect = lambda k: {'classifier': dummy_clf, 'preprocessor': fake_preprocessor}[k]
        pipeline_like.named_steps = {'classifier': dummy_clf, 'preprocessor': fake_preprocessor}

        # Set mock GridSearch-like results
        self.model._SurvivalPrediction__model_RF = MagicMock()
        self.model._SurvivalPrediction__model_RF.best_estimator_ = pipeline_like

        self.model._SurvivalPrediction__model_LR = MagicMock()
        self.model._SurvivalPrediction__model_LR.best_estimator_ = pipeline_like

        self.model._SurvivalPrediction__numerical_features = ['age', 'fare']
        self.model._SurvivalPrediction__categorical_features = ['sex', 'class', 'who']

        # These calls should not raise exceptions
        self.model.feature_importances()
        self.model.LR_feature_coeff()
        
if __name__ == '__main__':
    unittest.main()
