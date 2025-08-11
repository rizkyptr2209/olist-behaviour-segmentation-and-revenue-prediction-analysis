from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn. preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class RFMLBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data['first-purchase'] = pd.to_datetime(data['first-purchase'])
        data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
        data['order_delivered_customer_date'] = pd.to_datetime(data['order_delivered_customer_date'])

        # Add default customer ID if missing
        if 'customer_unique_id' not in data.columns:
            data['customer_unique_id'] = 'single_user'

        # Add missing columns with 0 or NaN
        for col in ['freight_value', 'price']:
          if col not in data.columns:
            data[col] = 0

        # Aggregate by customer
        agg = data.groupby('customer_unique_id').agg({
            'order_delivered_customer_date': 'max',
            'first-purchase': 'max',
            'order_purchase_timestamp': ['max', 'count'],
            'payment_value': 'sum',
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()

        agg.columns = ['customer_unique_id', 'last_interact', 'first_purchase', 'last_purchase', 'frequency',
                       'monetary', 'total_price', 'total_freight']

        # Use user-provided frequency if available
        if 'frequency' in data.columns and not data['frequency'].isnull().all():
            agg['Frequency'] = data['frequency'].iloc[0]
        else:
            agg['Frequency'] = agg['frequency']

        # RFML features
        reference_date = data['order_delivered_customer_date'].max()
        agg['Recency'] = (reference_date - agg['last_interact']).dt.days
        agg['Monetary'] = agg['monetary']
        agg['Loyalty'] = (agg['last_purchase'] - agg['first_purchase']).dt.days / agg['Frequency']

        # Target: total sales
        agg['total_sales'] = agg['total_price'] + agg['total_freight']

        # Drop intermediate columns
        agg.drop(columns=['last_interact', 'last_purchase', 'first_purchase', 'frequency', 'monetary', 'total_price', 'total_freight'], inplace=True)

        df_raw = agg.copy()

        return df_raw

class ClusterModel(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X[['Recency', 'Frequency', 'Monetary', 'Loyalty']])
        self.model.fit(X_scaled)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X[['Recency', 'Frequency', 'Monetary', 'Loyalty']])
        clusters = self.model.predict(X_scaled)  # Predict cluster labels
        X_out = X.copy()
        X_out['cluster_label'] = clusters
        return X_out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X[['Recency', 'Frequency', 'Monetary', 'Loyalty']])
        return self.model.predict(X_scaled)

class RFMLPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=5, rf_params=None, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.rf_params = rf_params if rf_params else {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': self.random_state
        }

        # Define scaler and Random Forest
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.model = RandomForestRegressor(**self.rf_params)

        # Preprocessor: scale RFML + pass cluster_label as-is
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scale', StandardScaler(), ['Recency', 'Frequency', 'Monetary', 'Loyalty']),
                ('passthrough', 'passthrough', ['cluster_label'])
            ]
        )

    def fit(self, X, y):
        # Fit GMM for cluster labeling
        X_scaled = self.scaler.fit_transform(X[['Recency', 'Frequency', 'Monetary', 'Loyalty']])
        clusters = self.gmm.fit_predict(X_scaled)

        # Add cluster label
        X_with_clusters = X.copy()
        X_with_clusters['cluster_label'] = clusters

        # Fit preprocessing + Random Forest
        X_processed = self.preprocessor.fit_transform(X_with_clusters)
        self.model.fit(X_processed, y)
        return self

    def predict(self, X):
        # Assign cluster labels using fitted GMM
        X_scaled = self.scaler.transform(X[['Recency', 'Frequency', 'Monetary', 'Loyalty']])
        clusters = self.gmm.predict(X_scaled)
        X_with_clusters = X.copy()
        X_with_clusters['cluster_label'] = clusters

        # Apply preprocessing + predict
        X_processed = self.preprocessor.transform(X_with_clusters)
        return self.model.predict(X_processed)