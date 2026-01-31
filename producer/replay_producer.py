import argparse
import json
import time
from datetime import datetime, timezone
import requests

from kafka import KafkaProducer

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{product}/candles"
# Coinbase returns candles: [ time, low, high, open, close, volume ]


def iso_to_unix(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.timestamp())


def unix_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def fetch_coinbase_candles(product: str, start_iso: str, end_iso: str, granularity: int = 60):
    params = {"start": start_iso, "end": end_iso, "granularity": granularity}
    url = COINBASE_CANDLES.format(product=product)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Coinbase returns newest-first; we want oldest-first
    data.sort(key=lambda x: x[0])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="BTC-USD", help="Coinbase product, e.g. BTC-USD")
    ap.add_argument("--start", required=True, help="e.g. 2026-01-18T18:00:00Z")
    ap.add_argument("--end", required=True, help="e.g. 2026-01-18T20:00:00Z")
    ap.add_argument("--bootstrap", default="localhost:19092")
    ap.add_argument("--topic", default="candles.btcusdt.1m")  # keep same topic name
    ap.add_argument("--speed", type=float, default=5.0, help="Replay speed multiplier")
    args = ap.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    candles = fetch_coinbase_candles(args.product, args.start, args.end, granularity=60)
    if not candles:
        print("No candles returned. Check time range.")
        return

    print(f"Replaying {len(candles)} candles from Coinbase {args.product} -> topic {args.topic} @ {args.speed}x")

    # For 1m candles, “real time” spacing is 60s; with speed=5, sleep is 12s
    sleep_s = max(0.0, 60.0 / args.speed)

    for (t, low, high, open_, close, volume) in candles:
        event = {
            "symbol": args.product,
            "open_time": t - 60,
            "close_time": t,
            "open_time_iso": unix_to_iso(t - 60),
            "close_time_iso": unix_to_iso(t),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
            "source": "coinbase_replay",
            "ingested_at_ms": int(time.time() * 1000),
        }
        producer.send(args.topic, value=event)

        # lightweight progress
        if (t % (60 * 10)) == 0:
            print("Sent candle close_time_iso:", event["close_time_iso"], "close:", event["close"])

        time.sleep(sleep_s)

    producer.flush()
    print("Done.")


if __name__ == "__main__":
    main()
