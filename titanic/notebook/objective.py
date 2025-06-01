import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

class Objective:
    def __init__(self, X, y, model, params, numerical, categorical, n_trial=100):
        self.X, self.y = X[numerical+categorical], y
        self.model, self.params = model, params
        self.numerical, self.categorical, self.n_trial = numerical, categorical, n_trial
        self.pbar = tqdm(total=n_trial)
        self.best_score, self.best_y_valid, self.best_y_pred, self.best_model, self.best_trial = None, None, None, None, None
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

    def __call__(self):
        self.study.optimize(self.objective, n_trials=self.n_trial)
        self.best_trial = self.study.best_trial
        self.pbar.close()
        
    def objective(self, trial):
        accuracies = []
        y_valids = []
        y_preds = []
        models = []
        sfk = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for train_idx, valid_idx in sfk.split(self.X, self.y):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]

            model = self.model(self.categorical)
            model.train(
                X_train, X_valid, y_train, y_valid, self.params(trial)
            )
            y_pred_prob = model.predict(X_valid)
            
            acc = accuracy_score(y_valid, (y_pred_prob > 0.5).astype(int))
            accuracies.append(acc)
            y_valids.extend(y_valid)
            y_preds.extend(y_pred_prob)
            models.append(model)
    
        if self.best_score is None or np.min(accuracies) > np.min(self.best_score):
            self.best_score = accuracies
            self.best_y_valid = np.array(y_valids)
            self.best_y_pred = np.array(y_preds)
            self.best_model = models
        self.pbar.update(1)
        return 1 - np.min(accuracies)




        
        
        