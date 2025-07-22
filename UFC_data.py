import pandas as pd

df = pd.read_csv('UFC_PPV_incomplete.csv')
df['Event'] = df['Event'].astype(str).str.strip()
full_events = pd.DataFrame({
    'Event': [f"UFC {i}" for i in range(173, 319)]  # 319 because upper bound is exclusive
})
df_full = pd.merge(full_events, df, on='Event', how='left')

# Optional: sort by event number to make sure it's ordered
df_full['Event_Num'] = df_full['Event'].str.extract(r'UFC (\d+)').astype(int)
df_full = df_full.sort_values('Event_Num').drop(columns='Event_Num')
df_full.to_csv("completed_ufc_events.csv", index=False)