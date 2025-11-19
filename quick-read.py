import pandas as pd

df = pd.read_parquet("results/app/eulogies/matches_scored.parquet")

# Sort by match probability (descending)
df_sorted = df.sort_values("match_proba", ascending=False)

# Save the full sorted file
df_sorted.to_parquet("results/app/eulogies/matches_scored_sorted.parquet", index=False)
df_sorted.to_csv("results/app/eulogies/matches_scored_sorted.csv", index=False)

print("Saved sorted matches to:")
print(" - matches_scored_sorted.parquet")
print(" - matches_scored_sorted.csv")