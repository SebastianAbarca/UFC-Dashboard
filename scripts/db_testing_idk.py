from scripts import load_data
import pandas as pd

df = load_data.load_data()
df_fighter_table = pd.read_csv('../data/FighterTable.csv')

print("df columns: ", df.columns)
print("df_fighter_table columns: ", df_fighter_table.columns)
