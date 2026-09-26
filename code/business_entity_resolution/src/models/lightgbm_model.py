import lightgbm as lgb
import numpy as np

BASE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def train_lightgbm(X_train, y_train, X_dev, y_dev, w_train=None, w_dev=None,
                   params: dict | None = None, n_estimators: int = 3000, early_stopping: int = 100):
    model = lgb.LGBMClassifier(n_estimators=n_estimators, **{**BASE_PARAMS, **(params or {})})
    model.fit(
        X_train, y_train, sample_weight=w_train,
        eval_set=[(X_dev, y_dev)], eval_sample_weight=[w_dev] if w_dev is not None else None,
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(200)],
    )
    return model


def importance(model, features: list[str]) -> list[dict]:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    rows = [{"feature": f, "gain": float(g), "split": int(s)} for f, g, s in zip(features, gain, split)]
    total = sum(r["gain"] for r in rows) or 1.0
    for r in rows:
        r["gain_pct"] = round(100 * r["gain"] / total, 3)
    return sorted(rows, key=lambda r: -r["gain"])


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float32)
