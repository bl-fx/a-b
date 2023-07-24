import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
import joblib
class DTCModel(object):  
    
    def __init__(self):
        
        self.categorical_features = [
            "person_home_ownership",
            "loan_intent",
            "city",
            "state",
            "location_type",
        ]
        
        self.encoder = joblib.load("encoder.pkl")
        
        print("Encoder loaded")
        
        self.model = joblib.load("DTC.pkl")
        
        print("Model loaded")
        
        self.cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        self.tries = 0
        self.success = 0
        self.value = 0
        

    def predict(self,X,features_names):

        df = pd.DataFrame(X, columns=features_names)
        
        df[self.categorical_features] = self.encoder.transform(df[self.categorical_features])
        df = df.reindex(sorted(df.columns), axis=1)
        
        predictions = self.model.predict_proba(df)
        
        return predictions

    def send_feedback(self, features, feature_names, reward, truth, routing=None):
        print("DTC model send-feedback entered")
        print(f"Truth: {truth}, Reward: {reward}")

        if reward == 1:
            if truth == 1:
                self.cm["tp"] += 1
            if truth == 0:
                self.cm["tn"] += 1
        if reward == 0:
            if truth == 1:
                self.cm["fn"] += 1
            if truth == 0:
                self.cm["fp"] += 1

        self.tries += 1
        self.success = self.success + 1 if reward else self.success
        self.value = self.success / self.tries

        print(self.cm)
        print(
            "Tries: %s, successes: %s, values: %s", self.tries, self.success, self.value
        )

    def metrics(self):
        tp = {
            "type": "GAUGE",
            "key": "true_pos_total",
            "value": self.cm["tp"],
            "tags": {"branch_name": "DTC"},
        }
        tn = {
            "type": "GAUGE",
            "key": "true_neg_total",
            "value": self.cm["tn"],
            "tags": {"branch_name": "DTC"},
        }
        fp = {
            "type": "GAUGE",
            "key": "false_pos_total",
            "value": self.cm["fp"],
            "tags": {"branch_name": "DTC"},
        }
        fn = {
            "type": "GAUGE",
            "key": "false_neg_total",
            "value": self.cm["fn"],
            "tags": {"branch_name": "DTC"},
        }

        value = {
            "type": "GAUGE",
            "key": "branch_value",
            "value": self.value,
            "tags": {"branch_name": "DTC"},
        }
        success = {
            "type": "GAUGE",
            "key": "n_success_total",
            "value": self.success,
            "tags": {"branch_name": "DTC"},
        }
        tries = {
            "type": "GAUGE",
            "key": "n_tries_total",
            "value": self.tries,
            "tags": {"branch_name": "DTC"},
        }

        return [tp, tn, fp, fn, value, success, tries]