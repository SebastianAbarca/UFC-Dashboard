import pandas as pd

df = pd.read_csv('../data/UFC_PPV_Tableau.csv')
df['Opponent1'] = df['Opponent1'].astype(str).str.strip().str.title()
df['Opponent2'] = df['Opponent2'].astype(str).str.strip().str.title()
fighters1 = df['Opponent1']
fighters2 = df['Opponent2']
concat_list = [fighters1, fighters2]
unique_fighters_concat = pd.concat(concat_list)
unique_fighters_concat = unique_fighters_concat.unique()
unique_fighters_concat = pd.DataFrame(unique_fighters_concat, columns=['Fighter'])
unique_fighters_concat.to_csv('unique_fighters.csv')