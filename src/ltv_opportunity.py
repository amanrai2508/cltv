import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class LTVOpportunityEstimator:
    def __init__(self, customer_data):
        """Initialize the LTV opportunity estimator.

        Args:
            customer_data (pd.DataFrame): Processed customer data from LTVDataPrep
        """
        self.data = customer_data.copy()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    def _prepare_features(self):
        """Prepare features for LTV prediction."""
        features = [
            'purchase_count',
            'avg_order_value',
            'customer_age_days',
            'purchase_frequency',
            'monthly_value'
        ]

        return self.data[features]

    def calculate_opportunities(self):
        """Calculate LTV opportunities for each customer."""
        # Prepare features
        X = self._prepare_features()
        y = self.data['total_amount']

        # Train model
        self.model.fit(X, y)

        # Predict potential LTV
        predicted_ltv = self.model.predict(X)

        # Add predictions to customer data
        results = self.data.copy()
        results['predicted_ltv'] = predicted_ltv
        results['opportunity_size'] = results['predicted_ltv'] - \
            results['total_amount']

        # Calculate confidence scores based on prediction reliability
        feature_importance = self.model.feature_importances_
        results['confidence_score'] = np.average(
            X.values,
            weights=feature_importance,
            axis=1
        )

        return results
