import pandas as pd

def load_data():
    df_ppv = pd.read_csv('ufc_events.csv')
    df_embedded = pd.read_csv('ufc_embedded_data.csv')
    df_ppv['PPV'] = df_ppv['PPV'].str.replace(',', '').replace('', pd.NA)
    df_ppv['PPV'] = pd.to_numeric(df_ppv['PPV'], errors='coerce')
    for col in df_embedded.columns[2:]:  # skip 'Unnamed: 0' and 'Event'
        df_embedded[col] = df_embedded[col].astype(str).str.replace(',', '').astype(float)
    df_embedded = df_embedded.drop(columns=['Unnamed: 0'])

    merged_df = pd.merge(df_ppv, df_embedded, on='Event', how='inner')

    # Add total YouTube views
    episode_cols = [col for col in merged_df.columns if 'Episode' in col]
    merged_df['Total_Embedded_Views'] = merged_df[episode_cols].sum(axis=1)

    return merged_df