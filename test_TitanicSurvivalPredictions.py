import unittest
import TitanicSurvivalPredictions

class TestTitanicSurvivalPredictions(unittest.TestCase):
    def test_split_data_raises_if_no_data():
        """Ensure split_data raises RuntimeError when data is missing."""
        model = TitanicSurvivalPrediction()
        with pytest.raises(RuntimeError):
            model.split_data()