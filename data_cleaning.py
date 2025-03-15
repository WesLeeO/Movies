import pandas as pd

# Load the dataset
file_path = "archive/movies_metadata.csv"  # Update this path if necessary
df = pd.read_csv(file_path, low_memory=False)

# Convert 'budget' and 'revenue' to numeric
df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

# Filter out rows where budget or revenue is zero or NaN
df_filtered = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

# Compute budget percentiles
low_budget_threshold = df_filtered["budget"].quantile(0.40)
high_budget_threshold = df_filtered["budget"].quantile(0.60)

# Define low-budget and high-budget movies
df_low_budget = df_filtered[df_filtered["budget"] <= low_budget_threshold]
df_high_budget = df_filtered[df_filtered["budget"] >= high_budget_threshold]

# Compute the revenue-to-budget multiplier
df_low_budget["multiplier"] = df_low_budget["revenue"] / df_low_budget["budget"]
df_high_budget["multiplier"] = df_high_budget["revenue"] / df_high_budget["budget"]

# Compute median multiplier for each category
median_multiplier_low = df_low_budget["multiplier"].median()
median_multiplier_high = df_high_budget["multiplier"].median()

# Split into two subcategories based on the multiplier median
df_low_budget_low_multiplier = df_low_budget[df_low_budget["multiplier"] <= median_multiplier_low]
df_low_budget_high_multiplier = df_low_budget[df_low_budget["multiplier"] > median_multiplier_low]

df_high_budget_low_multiplier = df_high_budget[df_high_budget["multiplier"] <= median_multiplier_high]
df_high_budget_high_multiplier = df_high_budget[df_high_budget["multiplier"] > median_multiplier_high]

# Save the six datasets
df_filtered.to_csv("archive/movies_metadata_filtered.csv", index=False)
df_low_budget.to_csv("archive/low_budget_movies.csv", index=False)
df_high_budget.to_csv("archive/high_budget_movies.csv", index=False)
df_low_budget_low_multiplier.to_csv("archive/low_budget_low_multiplier.csv", index=False)
df_low_budget_high_multiplier.to_csv("archive/low_budget_high_multiplier.csv", index=False)
df_high_budget_low_multiplier.to_csv("archive/high_budget_low_multiplier.csv", index=False)
df_high_budget_high_multiplier.to_csv("archive/high_budget_high_multiplier.csv", index=False)

print("Datasets successfully saved:")
print(f"  - movies_metadata_filtered.csv ({len(df_filtered)} rows)")
print(f"  - low_budget_movies.csv ({len(df_low_budget)} rows)")
print(f"  - high_budget_movies.csv ({len(df_high_budget)} rows)")
print(f"  - low_budget_low_multiplier.csv ({len(df_low_budget_low_multiplier)} rows)")
print(f"  - low_budget_high_multiplier.csv ({len(df_low_budget_high_multiplier)} rows)")
print(f"  - high_budget_low_multiplier.csv ({len(df_high_budget_low_multiplier)} rows)")
print(f"  - high_budget_high_multiplier.csv ({len(df_high_budget_high_multiplier)} rows)")
