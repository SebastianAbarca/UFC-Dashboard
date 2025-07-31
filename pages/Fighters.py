import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px
import streamlit as st
from scripts import load_data as ld

df = ld.load_data()
fighters_df = pd.concat([
    df[['Event', 'Date', 'Opponent1', 'PPV']].rename(columns={'Opponent1': 'Fighter'}),
    df[['Event', 'Date', 'Opponent2', 'PPV']].rename(columns={'Opponent2': 'Fighter'})
])
fighters_df['Date'] = pd.to_datetime(fighters_df['Date'])

st.header("1. Top Fighters by Total PPV")

fighter_ppv_totals = fighters_df.groupby('Fighter')['PPV'].sum().sort_values(ascending=False).head(15).reset_index()

fig1 = px.bar(
    fighter_ppv_totals,
    x='PPV',
    y='Fighter',
    orientation='h',
    title='Top 15 Fighters by Total PPV Impact',
    labels={'PPV': 'Total PPV Buys'},
    height=600
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("2. PPV by Event and Fighter")

selected_fighters = st.multiselect("Select Fighters to Compare", fighters_df['Fighter'].unique())

if selected_fighters:
    filtered = fighters_df[fighters_df['Fighter'].isin(selected_fighters)]
    fig2 = px.bar(
        filtered,
        x='Event',
        y='PPV',
        color='Fighter',
        title='PPV Buys by Event for Selected Fighters',
        labels={'PPV': 'PPV Buys'},
        height=600
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select at least one fighter to view event-wise PPV comparison.")

st.subheader("3. Fighter PPV Over Time")

fighter_choice = st.selectbox("Choose a Fighter", fighters_df['Fighter'].unique())

line_df = fighters_df[fighters_df['Fighter'] == fighter_choice].sort_values('Date')

fig3 = px.line(
    line_df,
    x='Date',
    y='PPV',
    hover_data=['Event'],
    title=f'PPV Trend Over Time for {fighter_choice}',
    markers=True,
    labels={'PPV': 'PPV Buys'}
)
st.plotly_chart(fig3, use_container_width=True)

# Machine Learning Section
df_ml = df.dropna(subset=['PPV', 'Opponent1', 'Opponent2'])
fighter_cols = ['Opponent1', 'Opponent2']

all_fighters = pd.unique(df_ml[fighter_cols].values.ravel())
all_fighters.sort()

X_fighters_full = pd.DataFrame(0, index=df_ml.index, columns=all_fighters)
for idx, row in df_ml.iterrows():
    X_fighters_full.at[idx, row['Opponent1']] = 1
    X_fighters_full.at[idx, row['Opponent2']] = 1

df_ml['Matchup'] = df_ml.apply(lambda row: ' vs '.join(sorted([row['Opponent1'], row['Opponent2']])), axis=1)

matchup_dummies_full = pd.get_dummies(df_ml['Matchup'])

y_full = df_ml['PPV'].astype(float)

# --- Core Logic for Fair Model Comparison ---

X_combined_full = pd.concat([X_fighters_full, matchup_dummies_full], axis=1)

X_fighters_only_full = X_fighters_full # This is already prepared above

X_train_combined, X_test_combined, y_train, y_test = train_test_split(
    X_combined_full, y_full, test_size=0.2, random_state=42
)

X_train_fighters_only = X_fighters_only_full.loc[X_train_combined.index]
X_test_fighters_only = X_fighters_only_full.loc[X_test_combined.index]

model_combined = Ridge(alpha=1.0)
model_combined.fit(X_train_combined, y_train)
y_pred_test_combined = model_combined.predict(X_test_combined)
r2_test_combined = r2_score(y_test, y_pred_test_combined)
st.write(f"Model R² score (Fighters + Matchups, Test Set): {r2_test_combined:.3f}")

influence = pd.Series(model_combined.coef_[:len(X_fighters_full.columns)], index=X_fighters_full.columns).sort_values(ascending=False)
matchup_influence = pd.Series(model_combined.coef_[len(X_fighters_full.columns):], index=matchup_dummies_full.columns).sort_values(ascending=False)


model_fighter_only = Ridge(alpha=1.0)
model_fighter_only.fit(X_train_fighters_only, y_train) # Fit on training data for fighter only
y_pred_test_fighter_only = model_fighter_only.predict(X_test_fighters_only) # Predict on test data for fighter only
r2_fighter_only_test = r2_score(y_test, y_pred_test_fighter_only) # Calculate R2 on test data
st.write(f"Model R² score (Fighters Only, Test Set): {r2_fighter_only_test:.3f}")


st.subheader("Estimated Fighter Influence on PPV (Regression Coefficients)")
top_fighters_df = influence.head(15).reset_index()
top_fighters_df.columns = ['Fighter', 'Coefficient']
fig = px.bar(
    top_fighters_df,
    x='Coefficient',
    y='Fighter',
    orientation='h',
    labels={'Coefficient': 'Estimated PPV Contribution'},
    title="Top 15 Fighters by Estimated PPV Influence (Regression)"
)
st.plotly_chart(fig)

st.subheader("Top Matchups by PPV Synergy")
top_matchups_df = matchup_influence.head(10).reset_index()
top_matchups_df.columns = ['Matchup', 'Coefficient']
fig_matchups = px.bar(
    top_matchups_df,
    x='Coefficient',
    y='Matchup',
    orientation='h',
    labels={'Coefficient': 'Synergistic PPV Contribution'},
    title="Top 10 Matchups by Estimated Synergy"
)
st.plotly_chart(fig_matchups)

df_ml_with_predictions = df_ml.copy()
df_ml_with_predictions['Predicted_PPV'] = pd.NA # Initialize with pandas NA

df_ml_with_predictions.loc[X_test_combined.index, 'Predicted_PPV'] = y_pred_test_combined

full_predictions = model_combined.predict(X_combined_full)
df_ml_with_predictions['Predicted_PPV_Full_Data'] = full_predictions
df_ml_with_predictions['Residual'] = df_ml_with_predictions['PPV'] - df_ml_with_predictions['Predicted_PPV_Full_Data']

st.subheader("Events that Over/Underperformed vs Model")

fig_scatter = px.scatter(
    df_ml_with_predictions, x='Predicted_PPV_Full_Data', y='PPV',
    hover_data=['Event', 'Opponent1', 'Opponent2', 'Residual'],
    title="Actual vs Predicted PPV Buys (Full Data, Model: Fighters + Matchups)",
    labels={'Predicted_PPV_Full_Data': 'Predicted PPV', 'PPV': 'Actual PPV'}
)
st.plotly_chart(fig_scatter)