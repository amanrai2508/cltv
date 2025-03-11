import pandas as pd
import numpy as np
from datetime import datetime


class LTVDataPrep:
    def __init__(self, transaction_data, customer_id_col='customer_id',
                 transaction_date_col='purchase_date', amount_col='amount'):
        """Initialize the LTV data preparation class.

        Args:
            transaction_data (pd.DataFrame): DataFrame containing transaction data
            customer_id_col (str): Name of customer ID column
            transaction_date_col (str): Name of transaction date column
            amount_col (str): Name of transaction amount column
        """
        self.data = transaction_data.copy()
        self.customer_id_col = customer_id_col
        self.transaction_date_col = transaction_date_col
        self.amount_col = amount_col

    def process_data(self):
        """Process transaction data for LTV analysis."""
        # Calculate key metrics per customer
        customer_metrics = self.data.groupby(self.customer_id_col).agg({
            self.transaction_date_col: ['min', 'max', 'count'],
            self.amount_col: ['sum', 'mean']
        }).reset_index()

        # Flatten column names
        customer_metrics.columns = [
            self.customer_id_col,
            'first_purchase_date',
            'last_purchase_date',
            'purchase_count',
            'total_amount',
            'avg_order_value'
        ]

        # Calculate time-based features
        customer_metrics['customer_age_days'] = (
            pd.to_datetime(customer_metrics['last_purchase_date']) -
            pd.to_datetime(customer_metrics['first_purchase_date'])
        ).dt.days

        customer_metrics['purchase_frequency'] = (
            customer_metrics['purchase_count'] /
            (customer_metrics['customer_age_days'] / 30)  # Monthly frequency
        ).fillna(0)

        # Calculate monetary value metrics
        customer_metrics['monthly_value'] = (
            customer_metrics['total_amount'] /
            (customer_metrics['customer_age_days'] / 30)
        ).fillna(0)

        return customer_metrics
