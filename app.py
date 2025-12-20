import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------ LOAD TRAINED MODEL AND SCALER ------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# ------------------ STREAMLIT APP CONFIG ------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload your transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # ------------------ FILTERS ------------------
    st.sidebar.subheader("Filter Transactions")

    if 'amount' in df.columns and 'hour' in df.columns and 'day_of_week' in df.columns:
        amount_range = st.sidebar.slider(
            "Transaction Amount",
            float(df['amount'].min()),
            float(df['amount'].max()),
            (float(df['amount'].min()), float(df['amount'].max()))
        )

        hour_range = st.sidebar.slider(
            "Transaction Hour",
            int(df['hour'].min()),
            int(df['hour'].max()),
            (int(df['hour'].min()), int(df['hour'].max()))
        )

        days_selected = st.sidebar.multiselect(
            "Day of Week",
            options=df['day_of_week'].unique(),
            default=list(df['day_of_week'].unique())
        )

        # Apply filters
        filtered_df = df[
            (df['amount'] >= amount_range[0]) &
            (df['amount'] <= amount_range[1]) &
            (df['hour'] >= hour_range[0]) &
            (df['hour'] <= hour_range[1]) &
            (df['day_of_week'].isin(days_selected))
        ].copy()
    else:
        filtered_df = df.copy()

    if len(filtered_df) == 0:
        st.warning("No transactions match the selected filters.")
    else:
        # ------------------ DATA PREPROCESSING ------------------

        # Drop unwanted/generated columns
        X = filtered_df.drop(columns=['transaction_id', 'is_fraud', 'fraud_probability'], errors='ignore')

        # Convert categorical columns to dummy variables (excluding transaction IDs)
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        cat_cols = [col for col in cat_cols if not col.lower().startswith('transaction')]
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Align columns with training data
        X = X.reindex(columns=model_columns, fill_value=0)

        # Scale numeric columns
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X[num_cols] = scaler.transform(X[num_cols])

        # ------------------ PREDICTION ------------------
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = model.predict(X)

        y_pred = (y_proba >= 0.5).astype(int)

        # Add results to dataframe
        filtered_df['Fraud_Probability'] = y_proba
        filtered_df['Predicted_Fraud'] = y_pred

        # ------------------ DISPLAY RESULTS ------------------
        st.subheader("Prediction Results")
        st.dataframe(filtered_df)

        st.subheader("⚠️ Predicted Fraud Transactions")
        st.dataframe(filtered_df[filtered_df['Predicted_Fraud'] == 1])

        # Summary statistics
        st.subheader("Transaction Summary")
        st.write("Total transactions:", len(filtered_df))
        st.write("Predicted frauds:", filtered_df['Predicted_Fraud'].sum())
        st.write("Average fraud probability:", round(filtered_df['Fraud_Probability'].mean(), 4))

        # Bar chart of fraud probabilities
        st.subheader("Fraud Probability Distribution")
        st.bar_chart(filtered_df['Fraud_Probability'])

        # ------------------ DOWNLOAD ------------------
        st.download_button(
            "Download Predicted Transactions",
            filtered_df.to_csv(index=False),
            file_name="predicted_transactions.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to get predictions.")
