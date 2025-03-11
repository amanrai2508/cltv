import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from src.ltv_utils import LTVDataPrep
from src.ltv_opportunity import LTVOpportunityEstimator


def generate_customer_data(n_customers=1000, start_date='2023-01-01', n_months=12):
    """Generate realistic customer transaction data with consistent purchasing patterns."""
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    data = []

    # Define customer segments with different behaviors
    segments = [
        {'name': 'high_value', 'weight': 0.2, 'purchase_freq': (
            15, 45), 'amount_range': (100, 500)},
        {'name': 'medium_value', 'weight': 0.5, 'purchase_freq': (
            30, 90), 'amount_range': (50, 200)},
        {'name': 'low_value', 'weight': 0.3, 'purchase_freq': (
            60, 180), 'amount_range': (20, 100)}
    ]

    # Create cohorts (monthly)
    for month in range(n_months):
        # Calculate cohort size with some randomness but trending upward
        base_size = n_customers // n_months
        growth_factor = 1 + (month * 0.1)  # 10% growth per month
        cohort_size = int(base_size * growth_factor * random.uniform(0.8, 1.2))

        cohort_date = start_date + timedelta(days=30*month)

        # Generate customers for this cohort
        for _ in range(cohort_size):
            # Select customer segment
            segment = random.choices(
                segments, weights=[s['weight'] for s in segments])[0]
            customer_id = f'CUST_{len(data):05d}'

            # Set acquisition date within the cohort month
            acquisition_date = cohort_date + \
                timedelta(days=random.randint(0, 29))

            # Generate multiple purchases over time
            last_purchase_date = acquisition_date
            purchase_count = random.randint(3, 12)  # Ensure multiple purchases

            for _ in range(purchase_count):
                # Add some randomness to purchase frequency
                days_between_purchases = random.randint(
                    segment['purchase_freq'][0],
                    segment['purchase_freq'][1]
                )
                purchase_date = last_purchase_date + \
                    timedelta(days=days_between_purchases)

                # Only include purchases within the analysis period
                if purchase_date <= start_date + timedelta(days=30*n_months):
                    amount = round(random.uniform(
                        segment['amount_range'][0],
                        segment['amount_range'][1]
                    ), 2)

                    data.append({
                        'customer_id': customer_id,
                        'acquisition_date': acquisition_date,
                        'purchase_date': purchase_date,
                        'amount': amount,
                        'cohort_month': cohort_date.strftime('%Y-%m'),
                        'segment': segment['name']
                    })

                last_purchase_date = purchase_date

    return pd.DataFrame(data)


def create_cohort_analysis(df):
    """Create detailed cohort analysis with retention and LTV metrics."""
    # Calculate customer lifetime value
    customer_ltv = df.groupby(['customer_id', 'cohort_month', 'acquisition_date']).agg({
        'amount': 'sum',
        'purchase_date': ['min', 'max', 'count']
    }).reset_index()

    # Flatten column names
    customer_ltv.columns = ['customer_id', 'cohort_month', 'acquisition_date',
                            'amount', 'first_purchase', 'last_purchase', 'purchase_count']

    # Calculate months since first purchase for each transaction
    df['cohort_month'] = pd.to_datetime(df['cohort_month'] + '-01')
    df['months_since_acquisition'] = ((df['purchase_date'].dt.year - df['acquisition_date'].dt.year) * 12 +
                                      (df['purchase_date'].dt.month - df['acquisition_date'].dt.month))

    # Create cohort matrix for actual LTV
    cohort_matrix = df.pivot_table(
        index='cohort_month',
        columns='months_since_acquisition',
        values='amount',
        aggfunc='sum'
    )

    # Calculate cumulative LTV
    for col in cohort_matrix.columns:
        if col > 0:
            cohort_matrix[col] = cohort_matrix[col] + cohort_matrix[col-1]

    # Calculate predicted LTV using LTVision
    ltv_prep = LTVDataPrep(
        transaction_data=df,
        customer_id_col='customer_id',
        transaction_date_col='purchase_date',
        amount_col='amount'
    )
    processed_data = ltv_prep.process_data()
    ltv_estimator = LTVOpportunityEstimator(processed_data)
    opportunities = ltv_estimator.calculate_opportunities()

    # Merge predicted LTV with customer data
    customer_data = df.groupby(['customer_id', 'cohort_month']).agg({
        'months_since_acquisition': 'max'
    }).reset_index()

    customer_predictions = pd.merge(
        customer_data,
        opportunities[['customer_id', 'predicted_ltv']],
        on='customer_id'
    )

    # Create predicted LTV cohort matrix
    predicted_matrix = customer_predictions.pivot_table(
        index='cohort_month',
        columns='months_since_acquisition',
        values='predicted_ltv',
        aggfunc='mean'
    )

    # Fill forward predicted values
    predicted_matrix = predicted_matrix.fillna(method='ffill', axis=1)

    # Format index
    cohort_matrix.index = cohort_matrix.index.strftime('%Y-%m')
    predicted_matrix.index = predicted_matrix.index.strftime('%Y-%m')

    return cohort_matrix, predicted_matrix, customer_ltv


def main():
    # Generate simulated data
    print("Generating simulated customer data...")
    df = generate_customer_data()

    # Create cohort analysis
    print("\nCreating cohort analysis...")
    cohort_matrix, predicted_matrix, customer_ltv = create_cohort_analysis(df)

    # Save results
    print("\nSaving results...")
    cohort_matrix.to_csv('cohort_actual_ltv.csv')
    predicted_matrix.to_csv('cohort_predicted_ltv.csv')

    print("\nActual LTV Cohort Analysis:")
    print(cohort_matrix.head())

    print("\nPredicted LTV Cohort Analysis:")
    print(predicted_matrix.head())


if __name__ == "__main__":
    main()
