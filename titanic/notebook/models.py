import pandas as pd

import lightgbm as lgb
class LGB:
    def __init__(self, categorical=[]):
        self.model = None
        self.categorical = categorical
    def train(self, X_train, X_valid, y_train, y_valid, params):
        self.model = lgb.train(
            params, 
            lgb.Dataset(X_train, label=y_train, categorical_feature=self.categorical), 
            valid_sets=[lgb.Dataset(X_valid, label=y_valid, categorical_feature=self.categorical)]
        )
        self.feature = X_train.columns
        return self.model
    def predict(self, X_test):
        return self.model.predict(X_test)
    def feature_importance(self, importance_type="gain"):
        importances = self.model.feature_importance(importance_type=importance_type)
        return pd.DataFrame({
            "feature": self.feature,
            "importance": importances
        })


from sklearn.ensemble import RandomForestClassifier
class RF:
    def __init__(self, categorical=[]):
        self.model = None
        self.categorical = categorical
    def train(self, X_train, X_valid, y_train, y_valid, params):
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        self.features = X_train.columns
        return self.model
    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
    def feature_importance(self):
        return pd.DataFrame({
            "feature": self.features,
            "importance": self.model.feature_importances_
        })

from catboost import CatBoostClassifier, Pool
class CAT:
    def __init__(self, categorical=[]):
        self.model = None
        self.categorical = categorical
    def train(self, X_train, X_valid, y_train, y_valid, params):
        cat_features = [X_train.columns.get_loc(col) for col in self.categorical]
        self.model = CatBoostClassifier(**params)
        self.model.fit(
            Pool(X_train, y_train, cat_features=cat_features), 
            eval_set=Pool(X_valid, y_valid, cat_features=cat_features)
        )
        self.features = X_train.columns
        return self.model
    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
    def feature_importance(self):
        importances = self.model.get_feature_importance()
        return pd.DataFrame({
            "feature": self.features,
            "importance": importances
        })

from sklearn.linear_model import LogisticRegression
class LR:
    def __init__(self, categorical=[]):
        self.model = None
        self.categorical = categorical
    def train(self, X_train, X_valid, y_train, y_valid, params):
        self.model = LogisticRegression(**params)
        self.model.fit(X_train, y_train)
        self.features = X_train.columns
        return self.model
    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]
    def feature_importance(self):
        coef = self.model.coef_.flatten()
        return pd.DataFrame({
            "feature": self.features,
            "importance": abs(coef)  # 絶対値を取って大きさで重要度
        })