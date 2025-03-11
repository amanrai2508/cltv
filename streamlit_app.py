import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ltv_cohort_analysis import generate_customer_data, create_cohort_analysis
from src.ltv_utils import LTVDataPrep
from src.ltv_opportunity import LTVOpportunityEstimator

# Set page config
st.set_page_config(page_title="LTV Cohort Analysis", layout="wide")

# Title and description
st.title("Customer Lifetime Value (LTV) Cohort Analysis")
st.write("This dashboard demonstrates cohort analysis and LTV predictions using simulated customer data.")

# Sidebar controls
st.sidebar.header("Parameters")
n_customers = st.sidebar.slider("Number of Customers", 100, 5000, 1000)
n_months = st.sidebar.slider("Number of Months", 3, 24, 12)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))

# Generate data
if st.sidebar.button("Generate New Data"):
    st.session_state.df = generate_customer_data(
        n_customers=n_customers,
        start_date=start_date.strftime("%Y-%m-%d"),
        n_months=n_months
    )

# Initialize or use existing data
if "df" not in st.session_state:
    st.session_state.df = generate_customer_data(
        n_customers=n_customers,
        start_date=start_date.strftime("%Y-%m-%d"),
        n_months=n_months
    )

df = st.session_state.df

# Display raw data sample
st.subheader("Sample Data")
st.dataframe(df.head())

# Create cohort analysis
cohort_matrix, predicted_matrix, customer_ltv = create_cohort_analysis(df)

# Combined view of actual and predicted LTV
st.subheader("Combined Actual and Predicted LTV Analysis")

# Create tabs for different views
tab1, tab2 = st.tabs(["Combined Matrix", "Opportunity Analysis"])

with tab1:
    # Create a combined matrix where predictions only show up for missing actuals
    combined_matrix = cohort_matrix.copy()

    # For each row (cohort), fill NaN values with predictions
    for idx in combined_matrix.index:
        row = combined_matrix.loc[idx]
        last_valid_idx = row.last_valid_index()
        if last_valid_idx is not None and last_valid_idx < row.index[-1]:
            # Fill only the values after the last actual value
            prediction_values = predicted_matrix.loc[idx, last_valid_idx+1:]
            combined_matrix.loc[idx, last_valid_idx+1:] = prediction_values

    # Create a mask for actual vs predicted values
    is_predicted = cohort_matrix.isna() & combined_matrix.notna()

    # Plot the combined matrix
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the heatmap
    sns.heatmap(combined_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)

    # Add markers for predicted values
    for i in range(is_predicted.shape[0]):
        for j in range(is_predicted.shape[1]):
            if is_predicted.iloc[i, j]:
                # Add a marker or pattern for predicted values
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor='blue', lw=2))

    plt.title('Combined Actual and Predicted LTV')
    plt.xlabel('Months Since Acquisition')
    plt.ylabel('Cohort Month')

    plt.tight_layout()
    st.pyplot(fig)

    # Add a description of the visualization
    st.markdown("""
    **Reading the Combined Matrix:**
    - Values without blue borders: Actual cumulative LTV
    - Values with blue borders: Predicted future LTV
    - Each row shows a cohort's progression
    - Predictions only appear after the last known actual value for each cohort
    """)

with tab2:
    st.subheader("LTV Opportunity Analysis")

    # Calculate opportunity size only for predicted values
    opportunity_matrix = pd.DataFrame(
        0, index=cohort_matrix.index, columns=cohort_matrix.columns)
    for idx in cohort_matrix.index:
        row = cohort_matrix.loc[idx]
        last_valid_idx = row.last_valid_index()
        if last_valid_idx is not None and last_valid_idx < row.index[-1]:
            # Calculate opportunity only for future months
            actual_value = row[last_valid_idx]
            future_predictions = predicted_matrix.loc[idx, last_valid_idx+1:]
            opportunity_matrix.loc[idx, last_valid_idx +
                                   1:] = future_predictions - actual_value

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.heatmap(opportunity_matrix, annot=True, fmt='.0f',
                cmap='RdYlGn', center=0, ax=ax3)
    plt.title('Future LTV Opportunity Size')
    plt.xlabel('Months Since Acquisition')
    plt.ylabel('Cohort Month')
    st.pyplot(fig3)

    # Add interpretation
    st.markdown("""
    **Interpreting the Opportunity Matrix:**
    - Shows potential future value compared to last known actual value
    - Green: Predicted growth in LTV
    - White: No change expected
    - Only shows opportunities for future months where we don't have actual data
    """)

# Display metrics
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df['customer_id'].unique()))

with col2:
    st.metric("Total Revenue", f"${df['amount'].sum():,.2f}")

with col3:
    avg_actual_ltv = cohort_matrix.iloc[:, -1].mean()
    st.metric("Average Actual LTV", f"${avg_actual_ltv:,.2f}")

with col4:
    avg_predicted_ltv = predicted_matrix.iloc[:, -1].mean()
    st.metric("Average Predicted LTV", f"${avg_predicted_ltv:,.2f}")

# Segment Analysis
st.subheader("Customer Segment Analysis")
segment_metrics = df.groupby('segment').agg({
    'customer_id': 'nunique',
    'amount': ['sum', 'mean'],
    'purchase_date': 'count'
}).round(2)

segment_metrics.columns = [
    'Unique Customers', 'Total Revenue', 'Avg Order Value', 'Number of Purchases']
st.dataframe(segment_metrics)

# Visualize segment distribution
fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(15, 5))

# Revenue by segment
segment_revenue = df.groupby('segment')['amount'].sum()
segment_revenue.plot(kind='pie', autopct='%1.1f%%', ax=ax4)
ax4.set_title('Revenue Distribution by Segment')

# Customer count by segment
segment_customers = df.groupby('segment')['customer_id'].nunique()
segment_customers.plot(kind='pie', autopct='%1.1f%%', ax=ax5)
ax5.set_title('Customer Distribution by Segment')

st.pyplot(fig4)
