import numpy as np
import xgboost as xgb

BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def train_xgboost(X_train, y_train, X_dev, y_dev, w_train=None, w_dev=None,
                  params: dict | None = None, n_estimators: int = 3000, early_stopping: int = 100):
    model = xgb.XGBClassifier(n_estimators=n_estimators, early_stopping_rounds=early_stopping,
                              **{**BASE_PARAMS, **(params or {})})
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_dev, y_dev)], sample_weight_eval_set=[w_dev] if w_dev is not None else None,
              verbose=200)
    return model


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)
