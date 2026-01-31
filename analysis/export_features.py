import json, csv, time
from kafka import KafkaConsumer

BOOTSTRAP = "localhost:19092"
TOPIC = "features.btcusdt.1m.v2"
OUTFILE = "features_sample.csv"
MAX_ROWS = 500  # enough for summary stats

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    consumer_timeout_ms=15000,
)

rows = []
for msg in consumer:
    v = msg.value
    # keep only rows with required numeric fields
    if all(k in v for k in ["log_ret", "rv_past", "rv_future", "vol_mean"]):
        rows.append({
            "timestamp": v.get("timestamp"),
            "log_return": float(v["log_ret"]),
            "rolling_volatility": float(v.get("rolling_volatility_5m", v["rv_past"])),  # fallback
            "past_volatility": float(v["rv_past"]),
            "future_volatility": float(v["rv_future"]),
        })
    if len(rows) >= MAX_ROWS:
        break

with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["timestamp", "log_return", "rolling_volatility", "past_volatility", "future_volatility"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Exported {len(rows)} feature rows to {OUTFILE}")
