import pandas as pd

df1 = pd.read_parquet("part1.parquet")
df2 = pd.read_parquet("part2.parquet")

combined = pd.concat([df1, df2], ignore_index=True)

combined.to_excel("combined.xlsx", index=False)

print("Done!")