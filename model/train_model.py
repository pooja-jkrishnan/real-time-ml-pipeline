import argparse
import json
import math
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


BINANCE_REST = "https://api.binance.com/api/v3/klines"


def iso_to_ms(s: str) -> int:
    # Accepts e.g. 2025-01-01T00:00:00Z
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    r = requests.get(BINANCE_REST, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def load_history(symbol: str, interval: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    start_ms = iso_to_ms(start_iso)
    end_ms = iso_to_ms(end_iso)

    rows = []
    cur = start_ms
    while cur < end_ms:
        batch = fetch_klines(symbol, interval, cur, end_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cur = int(batch[-1][6]) + 1  # next after close_time

    # Binance kline schema
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["close_time"] = df["close_time"].astype(np.int64)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df.sort_values("close_time").reset_index(drop=True)
    return df


def compute_features_and_label(df: pd.DataFrame, past_window: int, horizon: int) -> pd.DataFrame:
    # log return
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # past realized volatility over `past_window`
    df["rv_past"] = df["log_ret"].rolling(past_window).apply(lambda x: np.sqrt(np.sum(x * x)), raw=True)

    # volume rolling mean over `past_window`
    df["vol_mean"] = df["volume"].rolling(past_window).mean()

    # label: future realized volatility over next `horizon` minutes (1..horizon)
    future_returns = [df["log_ret"].shift(-i) for i in range(1, horizon + 1)]
    fut_mat = np.column_stack(future_returns)
    df["rv_future"] = np.sqrt(np.sum(np.square(fut_mat), axis=1))

    out = df.dropna().copy()
    return out


def time_order_split(X: np.ndarray, y: np.ndarray, test_frac: float):
    n = len(X)
    test_n = int(round(n * test_frac))
    test_n = max(1, test_n)
    split = n - test_n
    return X[:split], X[split:], y[:split], y[split:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", required=True, help="UTC like 2025-01-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="UTC like 2025-01-08T00:00:00Z")

    ap.add_argument("--past_window", type=int, default=5, help="minutes for rolling features")
    ap.add_argument("--horizon", type=int, default=5, help="minutes ahead for label")
    ap.add_argument("--test_frac", type=float, default=0.2)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--model_out", default="ridge_vol_model.joblib")

    # MLflow
    ap.add_argument("--mlflow_uri", default="http://localhost:5000")
    ap.add_argument("--experiment", default="offline-training-demo")
    ap.add_argument("--run_name", default="offline-ridge-vol-train")
    args = ap.parse_args()

    # --- Load data ---
    df = load_history(args.symbol, args.interval, args.start, args.end)
    data = compute_features_and_label(df, past_window=args.past_window, horizon=args.horizon)

    # Features: simple + interpretable
    X = data[["rv_past", "vol_mean", "log_ret"]].values
    y = data["rv_future"].values

    X_train, X_test, y_train, y_test = time_order_split(X, y, test_frac=args.test_frac)

    # --- Train ---
    model = Ridge(alpha=args.alpha)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(math.sqrt(mean_squared_error(y_test, preds)))

    # --- MLflow logging (guaranteed experiment selection) ---
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("symbol", args.symbol)
        mlflow.log_param("interval", args.interval)
        mlflow.log_param("start", args.start)
        mlflow.log_param("end", args.end)
        mlflow.log_param("past_window_min", args.past_window)
        mlflow.log_param("horizon_min", args.horizon)
        mlflow.log_param("test_frac", args.test_frac)
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("feature_names", json.dumps(["rv_past", "vol_mean", "log_ret"]))

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        joblib.dump(model, args.model_out)
        mlflow.log_artifact(args.model_out)

    print(f"Saved model -> {args.model_out}")
    print(f"Test MAE={mae:.6f} RMSE={rmse:.6f}")
    print(f"MLflow experiment='{args.experiment}' run_name='{args.run_name}' uri='{args.mlflow_uri}'")


if __name__ == "__main__":
    main()
