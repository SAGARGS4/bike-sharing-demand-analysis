"""
Feature Engineering Module
Prepares features for machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class FeatureEngineer:
    """Class for feature engineering and preprocessing"""
    
    def __init__(self, data_path='data/day.csv'):
        """Initialize feature engineer with data path"""
        self.data_path = Path(data_path)
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the bike sharing dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download day.csv from UCI Bike Sharing Dataset."
            )
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {len(self.df)} records")
        return self.df
    
    def create_features(self):
        """Create additional features from existing data"""
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()
        
        # Convert date to datetime
        df['dteday'] = pd.to_datetime(df['dteday'])
        
        # Extract date components
        df['year'] = df['dteday'].dt.year
        df['month'] = df['dteday'].dt.month
        df['day'] = df['dteday'].dt.day
        
        # Create interaction features
        df['temp_humidity'] = df['temp'] * df['hum']
        df['temp_windspeed'] = df['temp'] * df['windspeed']
        
        # Create weather quality score (inverse of weather situation)
        df['weather_quality'] = 5 - df['weathersit']
        
        # Is it a good day for biking? (temp > 0.5, low humidity < 0.7, good weather)
        df['good_biking_day'] = (
            (df['temp'] > 0.5) & 
            (df['hum'] < 0.7) & 
            (df['weathersit'] <= 2)
        ).astype(int)
        
        # Season indicators (one-hot encoding)
        df['is_spring'] = (df['season'] == 1).astype(int)
        df['is_summer'] = (df['season'] == 2).astype(int)
        df['is_fall'] = (df['season'] == 3).astype(int)
        df['is_winter'] = (df['season'] == 4).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
        
        print(f"Created features. New shape: {df.shape}")
        return df
    
    def prepare_features_for_modeling(self, target_col='cnt', 
                                     exclude_cols=None):
        """
        Prepare features and target for machine learning
        
        Args:
            target_col: Target variable column name
            exclude_cols: List of columns to exclude from features
        
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        if self.df is None:
            df = self.create_features()
        else:
            df = self.create_features()
        
        # Default columns to exclude
        if exclude_cols is None:
            exclude_cols = ['instant', 'dteday', 'casual', 'registered', 'cnt']
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"\nFeatures used: {list(X.columns)}")
        
        return X, y, list(X.columns)
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
        
        Returns:
            Scaled features
        """
        # Fit on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_importance_ready_data(self):
        """Get data ready for feature importance analysis"""
        X, y, feature_names = self.prepare_features_for_modeling()
        return X, y, feature_names


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.prepare_features_for_modeling()
    print(f"\nSample features:\n{X.head()}")
    print(f"\nSample target:\n{y.head()}")
