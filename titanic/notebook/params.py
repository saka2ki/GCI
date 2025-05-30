def lgb_params(trial):
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 5),  # 簡素化
        'num_leaves': trial.suggest_int('num_leaves', 7, 12),  # 葉数制限
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # 小さめで安定
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),  # サブサンプリング
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),  # 正則化を強めに
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 100),  # 過学習抑制
        'early_stopping_rounds': 10,
        'num_iterations': trial.suggest_int('n_iterations', 50, 200),
        'verbose': -1
    }
    return params

def rf_params(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 4, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 30),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1,
    }
    return params

def cat_params(trial):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'iterations': trial.suggest_int('iterations', 50, 300),
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 30,
        #'task_type': 'GPU',  # Uncomment if GPU is available
    }
    return params

def lr_params(trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    solver = 'saga'  # 全penaltyに対応
    params = {
        'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
        'penalty': penalty,
        'solver': solver,
        'max_iter': trial.suggest_int('max_iter', 6000, 10000),
        'random_state': 42
    }
    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
    return params