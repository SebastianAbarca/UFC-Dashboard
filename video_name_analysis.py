import pandas as pd
import re
# --- Regex to match UFC Embedded ---
pattern = r"UFC (\d+) Embedded: Vlog Series - Episode (\d+)"

df = pd.read_csv('ufc_video_data.csv')
filtered_rows = []
print(df.size)
print(df.columns)
print(df.head())
for row in df.itertuples():
    if re.match(pattern, row.Title):
        filtered_rows.append(row)

df_filtered = pd.DataFrame(filtered_rows)
df_filtered['Event'] = df_filtered['Event'].apply(lambda x: f"UFC {int(x)}" if pd.notna(x) else x)

# Then pivot your data (for rows where Event is not NaN)
df_pivot = df_filtered[df_filtered['Event'].notna()].pivot_table(
    index='Event',
    columns='Episode',
    values='Views',
    aggfunc='first'
)
df_pivot.columns = [f'Episode {int(col)}' for col in df_pivot.columns]
df_pivot = df_pivot.fillna(0)
df_pivot = df_pivot.reset_index()
df_pivot = df_pivot.to_csv('ufc_video_data_mergable.csv')