import pandas as pd

df_ppv_new = pd.read_csv('../data/ufc_events.csv')
df_ppv_old = pd.read_csv('../data/UFC_PPV_BuyRates.csv')

df_ppv_old = df_ppv_old.drop(columns=['PPV_match', 'Buyrate', 'Kaggle_PPV'])
df_ppv_old = df_ppv_old.rename(columns={'Final_PPV': 'PPV'})

df_ppv_new['PPV'] = df_ppv_new['PPV'].str.replace(',','', regex=False)
df_ppv_new['PPV'] = pd.to_numeric(df_ppv_new['PPV'], errors='coerce')
df_ppv_old['PPV'] = pd.to_numeric(df_ppv_old['PPV'], errors='coerce')

frames = [df_ppv_old, df_ppv_new]
result = pd.concat(frames)
result = result.reset_index()

result['Matchup'] = result.apply(lambda row: ' vs '.join(sorted([row['Opponent1'], row['Opponent2']])), axis=1)
result_sorted = result.sort_values(by=['Matchup', 'Date']).reset_index(drop=True)
result_sorted = result_sorted.drop_duplicates(subset=['Event', 'Date'], keep='first')
result_sorted['fight_number'] = result_sorted.groupby('Matchup').cumcount() + 1

result = result_sorted.copy()
result = result.sort_values(by=['Date']).reset_index(drop=True)
result.reset_index(inplace=True)
print(result.columns)

result.to_csv('UFC_PPV_Tableau.csv', index=False)