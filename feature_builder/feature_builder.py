import json
import time
import math
from collections import deque

import numpy as np
from kafka import KafkaConsumer, KafkaProducer

BOOTSTRAP = "localhost:19092"

INPUT_TOPIC = "candles.btcusdt.1m"
OUTPUT_TOPIC = "features.btcusdt.1m.v2"

PAST_WINDOW = 5   # minutes for rolling features
HORIZON = 5       # minutes ahead for label (future realized vol)


def safe_json_deserializer(m: bytes):
    """Return dict if JSON, else None (skip malformed records)."""
    try:
        return json.loads(m.decode("utf-8"))
    except Exception:
        return None


def _log_return(p_t: float, p_prev: float) -> float:
    return math.log(p_t / p_prev)


def _realized_vol(returns) -> float:
    arr = np.array(returns, dtype=float)
    return float(np.sqrt(np.sum(arr * arr)))


def main():
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=safe_json_deserializer,
        auto_offset_reset="earliest",           # backfill so v2 topic gets populated
        enable_auto_commit=True,
        group_id="feature-builder-v2-backfill", # new group => fresh offsets
    )

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    # Keep enough history to compute past-window + horizon label
    closes = deque(maxlen=PAST_WINDOW + HORIZON + 3)  # (ts_ms, ts_iso, close, volume)
    rets = deque(maxlen=PAST_WINDOW + HORIZON + 2)    # log returns aligned to closes

    print("✅ Feature builder v2 started -> outputs rv_past, vol_mean, log_ret, rv_future")
    print(f"   Input topic:  {INPUT_TOPIC}")
    print(f"   Output topic: {OUTPUT_TOPIC}")

    for msg in consumer:
        e = msg.value

        # 1) Skip malformed / non-JSON messages
        if e is None or not isinstance(e, dict):
            continue

        # 2) Skip non-candle messages (e.g., manual test events)
        if "close" not in e or "close_time" not in e or "close_time_iso" not in e:
            continue

        # 3) Parse candle fields safely
        try:
            close = float(e["close"])
            volume = float(e.get("volume", 0.0))
            ts_ms = int(e["close_time"])
            ts_iso = str(e["close_time_iso"])
        except Exception:
            continue

        # Append this candle
        closes.append((ts_ms, ts_iso, close, volume))

        # Compute most recent return
        if len(closes) < 2:
            continue

        prev_close = closes[-2][2]
        if prev_close <= 0 or close <= 0:
            continue

        rets.append(_log_return(close, prev_close))

        # Need enough returns to compute past-window RV and future RV label
        if len(rets) < (PAST_WINDOW + HORIZON):
            continue

        # Target candle = HORIZON minutes behind the newest candle
        idx_target_candle = -(HORIZON + 1)

        # Split past/future returns
        rets_list = list(rets)
        end_past = len(rets_list) - HORIZON

        past_slice = rets_list[end_past - PAST_WINDOW:end_past]
        fut_slice = rets_list[end_past:end_past + HORIZON]

        if len(past_slice) < PAST_WINDOW or len(fut_slice) < HORIZON:
            continue

        rv_past = _realized_vol(past_slice)
        rv_future = _realized_vol(fut_slice)

        # Volume mean over past window candles ending at target candle
        closes_list = list(closes)
        target_ts_ms, target_ts_iso, _, _ = closes_list[idx_target_candle]

        start_idx = idx_target_candle - (PAST_WINDOW - 1)
        past_candles = closes_list[start_idx: idx_target_candle + 1]
        if len(past_candles) < PAST_WINDOW:
            continue

        vol_mean = float(np.mean([x[3] for x in past_candles]))

        # Last log return in the past window
        log_ret = float(past_slice[-1])

        out = {
            "symbol": str(e.get("symbol", "BTCUSDT")),
            "timestamp": target_ts_iso,
            "source_event_time": target_ts_ms,

            # Features (match train_model.py)
            "rv_past": rv_past,
            "vol_mean": vol_mean,
            "log_ret": log_ret,

            # Label (for online evaluation)
            "rv_future": rv_future,

            "generated_at_ms": int(time.time() * 1000),
            "source": "feature_builder_v2",
        }

        producer.send(OUTPUT_TOPIC, value=out)
        print(f"Emitted v2 feature @ {out['timestamp']} rv_past={rv_past:.6f} rv_future={rv_future:.6f}")

    producer.flush()


if __name__ == "__main__":
    main()
