import streamlit as st
import load_data as ld
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Import the EarlyStopping callback from xgboost
from xgboost.callback import EarlyStopping # Add this line


## Data Handling
df = ld.load_data()

# --- CRITICAL FIX: Ensure 'Date' column in original df is datetime ---
# Use errors='coerce' to turn any unparseable dates into NaT (Not a Time),
# which pd.to_datetime can handle.
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


# --- IMPORTANT: Flag for Actual PPV before any modifications ---
df['Actual_PPV_Available'] = ~df['PPV'].isna()

fighters_df = pd.concat([
    df[['Event', 'Date', 'Opponent1', 'PPV']].rename(columns={'Opponent1': 'Fighter'}),
    df[['Event', 'Date', 'Opponent2', 'PPV']].rename(columns={'Opponent2': 'Fighter'})
])
# Ensure fighters_df 'Date' is also datetime (though main df fix should propagate)
fighters_df['Date'] = pd.to_datetime(fighters_df['Date'], errors='coerce')


## Containers for Streamlit UI
title_container = st.container()
general_container = st.container()
fighter_container = st.container()
ml_container = st.container()

## Streamlit App Layout
with title_container:
    st.title('UFC/Youtube Dashboard :martial_arts_uniform:')

with general_container:
    st.write("TODO: fill with general UFC/ML information")

with ml_container: # ML section
    st.header("PPV Prediction Model")  # Simplified header as only one model now

    # --- Data Preparation for XGBoost ---
    # Prepare df_ml for ML modeling (only rows with known PPV for training)
    # Ensure 'Date' column is datetime for proper sorting
    # df_ml will only contain rows where PPV is not NaN, so 'Actual_PPV_Available' will be True for all its rows
    df_ml = df[['Event', 'Opponent1', 'Opponent2', 'PPV', 'Total_Embedded_Views', 'Date']].copy()
    df_ml = df_ml.dropna(subset=['PPV', 'Opponent1', 'Opponent2'])
    df_ml['Total_Embedded_Views'] = df_ml['Total_Embedded_Views'].fillna(0)
    # This line might be redundant if df['Date'] is already converted, but harmless
    df_ml['Date'] = pd.to_datetime(df_ml['Date'], errors='coerce')  # Ensure Date is datetime

    # Create matchup pair key (sorted so "A vs B" == "B vs A")
    df_ml['Matchup'] = df_ml.apply(lambda row: ' vs '.join(sorted([row['Opponent1'], row['Opponent2']])), axis=1)

    # --- FEATURE ENGINEERING: fight_number ---
    # 1. Sort by Matchup and Date to ensure correct fight numbering sequence
    df_ml_sorted = df_ml.sort_values(by=['Matchup', 'Date']).reset_index(drop=True)

    # 2. Group by 'Matchup' and use cumcount() to get the fight number
    # Add 1 to make it 1-based (first fight is 1, second is 2, etc.)
    df_ml_sorted['fight_number'] = df_ml_sorted.groupby('Matchup').cumcount() + 1

    # Replace original df_ml with the sorted and feature-engineered version
    df_ml = df_ml_sorted.copy()

    # Define features and target for XGBoost
    categorical_features_ohe = ['Opponent1', 'Opponent2', 'Matchup']
    numerical_features = ['Total_Embedded_Views', 'fight_number']  # 'fight_number' included here

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_ohe),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='passthrough'
    )

    # Apply preprocessing to create X for XGBoost
    # Fit the preprocessor on the *entire* df_ml before splitting to ensure
    # all possible categories are learned.
    X_ohe = preprocessor.fit_transform(df_ml[categorical_features_ohe + numerical_features])

    # Get feature names after OHE for XGBoost feature importance
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_ohe).tolist()
    all_ohe_feature_names = ohe_feature_names + numerical_features

    # Convert X_ohe to a DataFrame for easier handling, especially for feature importance later
    X_ohe_df = pd.DataFrame(X_ohe.toarray(), columns=all_ohe_feature_names, index=df_ml.index)
    y_ohe = df_ml['PPV'].astype(float)  # Target remains the same

    # --- Split data for XGBoost ---
    indices = df_ml.index
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    X_train_ohe, X_test_ohe = X_ohe_df.loc[train_indices], X_ohe_df.loc[test_indices]
    y_train_ohe, y_test_ohe = y_ohe.loc[train_indices], y_ohe.loc[test_indices]

    st.subheader("XGBoost Regressor")

    model_xgb = XGBRegressor(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        tree_method='hist',
        objective='reg:squarederror',
        eval_metric='rmse',  # Monitor RMSE for early stopping
        early_stopping_rounds=50,
        verbose=False
    )

    st.info("Training XGBoost model...")
    model_xgb.fit(X_train_ohe, y_train_ohe,
                  eval_set=[(X_test_ohe, y_test_ohe)])

    y_pred_xgb = model_xgb.predict(X_test_ohe)
    r2_xgb = r2_score(y_test_ohe, y_pred_xgb)
    st.write(f"**XGBoost Model R² score (Test Set): {r2_xgb:.3f}**")

    # Get feature importance for XGBoost
    feature_importances_xgb = pd.DataFrame({
        'Feature Id': all_ohe_feature_names,
        'Importances': model_xgb.feature_importances_
    }).sort_values(by='Importances', ascending=False)

    st.write("XGBoost Top 10 Feature Importance:")
    fig_importance_xgb = px.bar(
        feature_importances_xgb.head(10),
        x='Importances',
        y='Feature Id',
        orientation='h',
        title='Top 10 Most Important Features (XGBoost)'
    )
    st.plotly_chart(fig_importance_xgb, use_container_width=True)

    st.markdown("---")  # Separator for visualization section

    # --- Plotting Actual vs Predicted for XGBoost ---
    df_ml_with_predictions_xgb = df_ml.copy()
    df_ml_with_predictions_xgb['Predicted_PPV'] = model_xgb.predict(X_ohe_df)
    df_ml_with_predictions_xgb['Residual'] = df_ml_with_predictions_xgb['PPV'] - df_ml_with_predictions_xgb[
        'Predicted_PPV']

    st.subheader("XGBoost: Actual vs Predicted PPV Buys")
    fig_scatter_xgb = px.scatter(
        df_ml_with_predictions_xgb,
        x='Predicted_PPV',
        y='PPV',
        hover_data=['Event', 'Opponent1', 'Opponent2', 'Residual', 'Total_Embedded_Views', 'fight_number'],
        title="Actual vs Predicted PPV Buys (XGBoost Model)",
        labels={'Predicted_PPV': 'Predicted PPV Buys', 'PPV': 'Actual PPV Buys'},
        template="plotly_white"
    )
    fig_scatter_xgb.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
    fig_scatter_xgb.update_layout(showlegend=False, xaxis_title="Predicted PPV Buys", yaxis_title="Actual PPV Buys",
                                  shapes=[
                                      dict(
                                          type='line',
                                          xref='paper', yref='paper',
                                          x0=0, y0=0, x1=1, y1=1,
                                          line=dict(color='red', width=2, dash='dash')
                                      )
                                  ])
    st.plotly_chart(fig_scatter_xgb, use_container_width=True)

    st.write("---")
    st.write("### XGBoost: Residual Analysis (Events with Largest Over/Under Predictions)")

    df_ml_with_predictions_xgb['Abs_Residual'] = abs(df_ml_with_predictions_xgb['Residual'])
    df_sorted_residuals_xgb = df_ml_with_predictions_xgb.sort_values(by='Abs_Residual', ascending=False).head(15)

    fig_residuals_xgb = px.bar(
        df_sorted_residuals_xgb,
        x='Event',
        y='Residual',
        color='Residual',
        color_continuous_scale=px.colors.sequential.RdBu,
        title="Top 15 Events by Prediction Residual (XGBoost)",
        labels={'Residual': 'Actual PPV - Predicted PPV'},
        hover_data=['Opponent1', 'Opponent2', 'PPV', 'Predicted_PPV', 'Total_Embedded_Views', 'fight_number'],
        orientation='v'
    )
    fig_residuals_xgb.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_residuals_xgb, use_container_width=True)

    st.header("Predicting Missing PPV Values in Original Data")

    # Identify rows in the original 'df' where 'PPV' is missing
    df_missing_ppv = df[df['PPV'].isna()].copy()

    if not df_missing_ppv.empty:
        st.write(f"Found {len(df_missing_ppv)} events with missing PPV values to predict.")

        # Prepare data for prediction, mirroring the feature engineering from training
        df_predict = df_missing_ppv[['Event', 'Opponent1', 'Opponent2', 'Total_Embedded_Views', 'Date']].copy()
        df_predict['Total_Embedded_Views'] = df_predict['Total_Embedded_Views'].fillna(0)  # Fill NaNs for prediction
        # This line might be redundant if df['Date'] is already converted, but harmless
        df_predict['Date'] = pd.to_datetime(df_predict['Date'], errors='coerce')

        # Create matchup key for prediction data
        df_predict['Matchup'] = df_predict.apply(lambda row: ' vs '.join(sorted([row['Opponent1'], row['Opponent2']])),
                                                 axis=1)

        # --- FEATURE ENGINEERING: fight_number for prediction data ---
        # To calculate fight_number correctly for prediction data, we need to consider all historical fights.
        # Use the full 'df' (with the 'Date' column now consistently datetime)
        df_all_events_for_fight_num = df[['Event', 'Date', 'Opponent1', 'Opponent2']].copy()
        # Ensure 'Date' is datetime here as well, although it should be from the initial fix
        df_all_events_for_fight_num['Date'] = pd.to_datetime(df_all_events_for_fight_num['Date'], errors='coerce')
        df_all_events_for_fight_num['Matchup'] = df_all_events_for_fight_num.apply(
            lambda row: ' vs '.join(sorted([row['Opponent1'], row['Opponent2']])), axis=1)

        df_all_events_for_fight_num_sorted = df_all_events_for_fight_num.sort_values(
            by=['Matchup', 'Date']).reset_index(drop=True)
        df_all_events_for_fight_num_sorted['fight_number'] = df_all_events_for_fight_num_sorted.groupby(
            'Matchup').cumcount() + 1

        # Merge the calculated fight_number back to df_predict using the original index or a common key
        df_predict = df_predict.merge(
            df_all_events_for_fight_num_sorted[['Event', 'Matchup', 'fight_number']],
            on=['Event', 'Matchup'],
            how='left',
            # Removed suffixes as they are not needed if column names are distinct
        )

        # Select features for preprocessing (same as training)
        X_predict_raw = df_predict[categorical_features_ohe + numerical_features]

        # Use the *already fitted* preprocessor to transform the prediction data
        X_predict_ohe = preprocessor.transform(X_predict_raw)
        X_predict_ohe_df = pd.DataFrame(X_predict_ohe.toarray(), columns=all_ohe_feature_names, index=df_predict.index)

        # Make predictions
        predicted_ppv_values = model_xgb.predict(X_predict_ohe_df)

        # Add predictions to the df_predict DataFrame
        df_predict['Predicted_PPV'] = predicted_ppv_values

        st.subheader("Predicted PPV Values for Missing Events:")
        st.dataframe(
            df_predict[['Event', 'Opponent1', 'Opponent2', 'Predicted_PPV',
                        'fight_number']])  # Added fight_number for debug/info

        # Update the original 'df' with the predicted values
        # This will iterate through the indices of df_predict and update the corresponding PPV in df
        for idx, row in df_predict.iterrows():
            df.loc[idx, 'PPV'] = row['Predicted_PPV']

        st.success("Missing PPV values have been predicted and imputed into the DataFrame!")

        st.subheader("All UFC Events: Actual vs. Predicted PPV Buys")

        # Create a new DataFrame for plotting
        # For actual values, we'll use the original 'PPV' and mark them as 'Actual'
        # Filter df based on the 'Actual_PPV_Available' flag created at the beginning
        df_actual_ppv = df[df['Actual_PPV_Available']].copy()
        df_actual_ppv['PPV_Type'] = 'Actual'
        df_actual_ppv['Display_PPV'] = df_actual_ppv['PPV']

        # For predicted values, we'll use the 'Predicted_PPV' from df_predict and mark them as 'Predicted'
        # We need to ensure that the df_predict used here still has the original index from df_missing_ppv
        # df_predict already contains 'Predicted_PPV' and 'Event', 'Opponent1', 'Opponent2', 'Date'
        df_predicted_ppv_for_plot = df_predict.copy()
        df_predicted_ppv_for_plot['PPV_Type'] = 'Predicted'
        df_predicted_ppv_for_plot['Display_PPV'] = df_predicted_ppv_for_plot['Predicted_PPV']


        # Combine these two dataframes
        df_combined_plot = pd.concat([
            df_actual_ppv[['Event', 'Date', 'Opponent1', 'Opponent2', 'Display_PPV', 'PPV_Type']],
            df_predicted_ppv_for_plot[['Event', 'Date', 'Opponent1', 'Opponent2', 'Display_PPV', 'PPV_Type']]
        ])

        # Sort by date for better visualization (optional but often good)
        # The 'Date' column in df_combined_plot should now be consistently datetime objects
        df_combined_plot = df_combined_plot.sort_values(by='Date').reset_index(drop=True)

        fig_combined_ppv = px.bar(
            df_combined_plot,
            x='Event',
            y='Display_PPV',
            color='PPV_Type',  # This is what creates different colors
            barmode='group',  # Changed to 'group' for better distinction in case of overlap
            title='UFC Event PPV Buys: Actual vs. Predicted',
            labels={'Display_PPV': 'PPV Buys', 'Event': 'UFC Event'},
            hover_data=['Opponent1', 'Opponent2', 'Date'],
            color_discrete_map={'Actual': 'blue', 'Predicted': 'orange'}  # Customize colors
        )
        fig_combined_ppv.update_layout(xaxis_tickangle=-45)  # Rotate event names for readability
        st.plotly_chart(fig_combined_ppv, use_container_width=True)
