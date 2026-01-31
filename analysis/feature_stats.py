import pandas as pd

df = pd.read_csv("features_sample.csv")

cols = ["log_return", "rolling_volatility", "past_volatility", "future_volatility"]
stats = df[cols].agg(["mean", "std", "min", "max"]).T
stats = stats.rename(columns={"std": "std_dev"})

print("\nSummary statistics:\n")
print(stats)

stats.to_csv("table4_2_stats.csv")
print("\nSaved -> table4_2_stats.csv")
