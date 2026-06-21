import joblib
import numpy as np

class Predictor:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = joblib.load(model_path)

    def predict(self, features):
        pass

    def predict_proba(self, features):
        pass

if __name__ == "__main__":
    pass
