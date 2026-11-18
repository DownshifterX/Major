import pandas as pd

df = pd.read_excel('land_disputes.xlsx', nrows=3)
print("COLUMNS:", df.columns.tolist())
print()
print(df.head(2).to_string())
