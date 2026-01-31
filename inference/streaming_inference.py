import time
import json
from collections import deque

import joblib
import mlflow
import numpy as np
from kafka import KafkaConsumer


# ======================
# Configuration
# ======================
BOOTSTRAP_SERVERS = "localhost:19092"          # host port mapped to Kafka
FEATURE_TOPIC = "features.btcusdt.1m.v2"       # v2 topic (aligned schema)
GROUP_ID = "inference-consumer-v2"

MODEL_PATH = "../model/ridge_vol_model.joblib"

ROLLING_WINDOW = 50     # rolling window for MAE/RMSE
LOG_EVERY = 10          # log/print every N processed events

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "realtime-inference-demo"
RUN_NAME = "realtime-streaming-inference-v2"


# ======================
# Helpers
# ======================
def safe_json_deserializer(m: bytes):
    try:
        return json.loads(m.decode("utf-8"))
    except Exception:
        return None


def main():
    # ----------------------
    # Load model
    # ----------------------
    model = joblib.load(MODEL_PATH)
    print("Model loaded")

    # ----------------------
    # MLflow setup
    # ----------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ----------------------
    # Kafka consumer
    # ----------------------
    consumer = KafkaConsumer(
        FEATURE_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=safe_json_deserializer,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=GROUP_ID,
    )

    # ----------------------
    # Rolling buffers
    # ----------------------
    y_true_buf = deque(maxlen=ROLLING_WINDOW)
    y_pred_buf = deque(maxlen=ROLLING_WINDOW)

    counter = 0
    start_time = time.time()

    # ----------------------
    # Start run
    # ----------------------
    with mlflow.start_run(run_name=RUN_NAME):
        # Log config params once
        mlflow.log_param("feature_topic", FEATURE_TOPIC)
        mlflow.log_param("rolling_window", ROLLING_WINDOW)
        mlflow.log_param("log_every", LOG_EVERY)
        mlflow.log_param("model_path", MODEL_PATH)

        print(f"Consuming from topic: {FEATURE_TOPIC}")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")

        for msg in consumer:
            event = msg.value

            # Skip malformed messages
            if event is None or not isinstance(event, dict):
                continue

            # Expect v2 schema
            required = ["rv_past", "vol_mean", "log_ret", "rv_future"]
            if not all(k in event for k in required):
                continue

            # Build feature vector
            X = np.array([
                float(event["rv_past"]),
                float(event["vol_mean"]),
                float(event["log_ret"]),
            ]).reshape(1, -1)

            # True label (delayed, realized)
            y_true = float(event["rv_future"])

            # Inference latency (model predict only)
            t0 = time.time()
            y_pred = float(model.predict(X)[0])
            latency_ms = (time.time() - t0) * 1000.0

            # Update rolling buffers
            y_true_buf.append(y_true)
            y_pred_buf.append(y_pred)
            counter += 1

            # Compute rolling metrics when buffer has enough values
            if len(y_true_buf) >= 5:
                y_true_arr = np.array(y_true_buf, dtype=float)
                y_pred_arr = np.array(y_pred_buf, dtype=float)

                mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
                rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

                # Log + print periodically
                if counter % LOG_EVERY == 0:
                    elapsed = time.time() - start_time
                    throughput = counter / elapsed if elapsed > 0 else 0.0

                    # Log to MLflow
                    mlflow.log_metric("online_mae", mae, step=counter)
                    mlflow.log_metric("online_rmse", rmse, step=counter)
                    mlflow.log_metric("latency_ms", latency_ms, step=counter)
                    mlflow.log_metric("throughput_eps", throughput, step=counter)

                    # Console output (for screenshots)
                    ts = event.get("timestamp", "n/a")
                    print(
                        f"[{counter}] ts={ts} "
                        f"MAE={mae:.6f} RMSE={rmse:.6f} "
                        f"latency={latency_ms:.2f}ms "
                        f"throughput={throughput:.2f}/s"
                    )


if __name__ == "__main__":
    main()
