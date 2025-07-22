import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import load_data as ld
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

df = ld.load_data()
st.title('Predicting PPV Buys from UFC Embedded Videos')
st.markdown('---')

episodes_cols = [f'Episode {i}' for i in range(1, 10)]

st.subheader("Select an Event to See Embedded Episode Views")

# Event selector
selected_event = st.selectbox("Choose an Event", df['Event'].unique())

if selected_event:
    event_data = df[df['Event'] == selected_event][episodes_cols].T.reset_index()
    event_data.columns = ['Episode', 'Views']

    fig = px.bar(
        event_data,
        x='Episode',
        y='Views',
        title=f'Embedded Episode Views for {selected_event}',
        labels={'Views': 'Views', 'Episode': 'Episode'}
    )
    st.plotly_chart(fig, use_container_width=True)
st.markdown('---')

fig = px.scatter(
    df,
    x='PPV',
    y='Total_Embedded_Views',
    hover_data=['Event', 'Opponent1', 'Opponent2'],
    labels={
        'Total_Embedded_Views': 'YouTube Views',
        'PPV': 'PPV Buys'
    },
    title='Total YouTube Embedded Views vs PPV Buys'
)
st.plotly_chart(fig, use_container_width=True)

ep_melted = df.melt(id_vars=['Event', 'Date', 'PPV'], value_vars=episodes_cols, var_name='Episode', value_name='Views')

y_max = np.percentile(ep_melted['Views'], 99.78)
fig_ep_views = px.box(
    ep_melted,
    x='Episode',
    y='Views',
    title='Distribution of Embedded Views by Episode',
    labels={'Views': 'Views'}
)

fig_ep_views.update_yaxes(range=[0, y_max])
st.plotly_chart(fig_ep_views, use_container_width=True)

correlation = df['PPV'].corr(df['Total_Embedded_Views'], method='pearson')
st.write("Pearson Correlation between Total Views and PPV buys", correlation)

df[episodes_cols] = df[episodes_cols].fillna(0)
df_model = df.dropna(subset=['PPV'])

# ============ Multivariate Ridge Regression with Episodes ============
X = df_model[episodes_cols]  # Using episodes as features
y = df_model['PPV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge_pipeline.fit(X_train, y_train)

y_pred_ridge = ridge_pipeline.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
cv_scores = cross_val_score(ridge_pipeline, X, y, cv=5, scoring='r2')

st.subheader("Multivariate Ridge Regression (Episodes)")
st.write(f"Cross-validated R² scores: {cv_scores}")
st.write(f"Mean CV R²: {cv_scores.mean():.3f}")
st.write(f"R² on test set: {r2_ridge:.3f}")
st.write(f"Mean Squared Error on test set: {mse_ridge:.3f}")

# ============ Univariate Ridge Regression with Total Embedded Views ============
df_total = df_model.dropna(subset=['Total_Embedded_Views'])  # Ensure no NaNs here

X_total = df_total[['Total_Embedded_Views']]
y_total = df_total['PPV']

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_total, y_total, test_size=0.2, random_state=42)

ridge_uni_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge_uni_pipeline.fit(X_train_t, y_train_t)

y_pred_uni = ridge_uni_pipeline.predict(X_test_t)
r2_uni = r2_score(y_test_t, y_pred_uni)
mse_uni = mean_squared_error(y_test_t, y_pred_uni)
cv_scores_uni = cross_val_score(ridge_uni_pipeline, X_total, y_total, cv=5, scoring='r2')

st.subheader("Univariate Ridge Regression (Total Embedded Views)")
st.write(f"Cross-validated R² scores: {cv_scores_uni}")
st.write(f"Mean CV R²: {cv_scores_uni.mean():.3f}")
st.write(f"R² on test set: {r2_uni:.3f}")
st.write(f"Mean Squared Error on test set: {mse_uni:.3f}")

# Scatter plot Actual vs Predicted for univariate model
fig_uni = px.scatter(
    x=y_test_t,
    y=y_pred_uni,
    labels={'x': 'Actual PPV', 'y': 'Predicted PPV'},
    title='Actual vs Predicted PPV (Univariate Ridge)'
)
fig_uni.add_shape(
    dict(type='line', line=dict(dash='dash'), x0=y_test_t.min(), x1=y_test_t.max(), y0=y_test_t.min(), y1=y_test_t.max())
)
st.plotly_chart(fig_uni, use_container_width=True)
