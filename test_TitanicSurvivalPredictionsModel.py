import unittest
from unittest.mock import patch, MagicMock
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

        from sklearn.compose import ColumnTransformer
        self.assertIsInstance(self.model._SurvivalPrediction__preprocessor, ColumnTransformer)

if __name__ == '__main__':
    unittest.main()
