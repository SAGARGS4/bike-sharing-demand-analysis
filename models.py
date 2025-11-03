"""
Machine Learning Models Module
Implements various ML models for bike demand prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path


class BikeRentalPredictor:
    """Class for training and evaluating bike rental prediction models"""
    
    def __init__(self):
        """Initialize predictor with model dictionary"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        }
        self.results = {}
        self.trained_models = {}
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            random_state: Random seed
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
        
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        print(f"{model_name} trained successfully")
        
        return model
    
    def evaluate_model(self, model_name, model, X_train, X_test, y_train, y_test):
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        self.results[model_name] = metrics
        
        print(f"\n{model_name} Results:")
        print(f"  Training R²: {metrics['train_r2']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Training RMSE: {metrics['train_rmse']:.2f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"  Training MAE: {metrics['train_mae']:.2f}")
        print(f"  Test MAE: {metrics['test_mae']:.2f}")
        
        return metrics
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        """
        print("\n" + "="*50)
        print("Training All Models")
        print("="*50)
        
        for model_name in self.models.keys():
            model = self.train_model(model_name, X_train, y_train)
            self.evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        
        print("\n" + "="*50)
        print("All models trained and evaluated")
        print("="*50)
    
    def plot_model_comparison(self, save_path='plots'):
        """Create visualization comparing all models"""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        Path(save_path).mkdir(exist_ok=True)
        
        # Prepare data for plotting
        models = list(self.results.keys())
        test_r2 = [self.results[m]['test_r2'] for m in models]
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        test_mae = [self.results[m]['test_mae'] for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # R² Score comparison
        axes[0].bar(models, test_r2, color=['skyblue', 'lightgreen', 'coral', 'plum'])
        axes[0].set_title('R² Score (Test Set)')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[1].bar(models, test_rmse, color=['skyblue', 'lightgreen', 'coral', 'plum'])
        axes[1].set_title('RMSE (Test Set)')
        axes[1].set_ylabel('RMSE')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[2].bar(models, test_mae, color=['skyblue', 'lightgreen', 'coral', 'plum'])
        axes[2].set_title('MAE (Test Set)')
        axes[2].set_ylabel('MAE')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nModel comparison plot saved to {save_path}/model_comparison.png")
        plt.close()
    
    def plot_predictions_vs_actual(self, X_test, y_test, save_path='plots'):
        """Plot predicted vs actual values for all models"""
        if not self.trained_models:
            print("No trained models. Train models first.")
            return
        
        Path(save_path).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Predictions vs Actual Values', fontsize=16)
        axes = axes.ravel()
        
        for idx, (model_name, model) in enumerate(self.trained_models.items()):
            y_pred = model.predict(X_test)
            
            axes[idx].scatter(y_test, y_pred, alpha=0.5)
            axes[idx].plot([y_test.min(), y_test.max()], 
                          [y_test.min(), y_test.max()], 
                          'r--', lw=2, label='Perfect Prediction')
            axes[idx].set_xlabel('Actual Values')
            axes[idx].set_ylabel('Predicted Values')
            axes[idx].set_title(f'{model_name}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Add R² score on plot
            r2 = self.results[model_name]['test_r2']
            axes[idx].text(0.05, 0.95, f'R² = {r2:.4f}', 
                          transform=axes[idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print(f"Predictions vs actual plot saved to {save_path}/predictions_vs_actual.png")
        plt.close()
    
    def get_feature_importance(self, feature_names, save_path='plots'):
        """Get feature importance from tree-based models"""
        Path(save_path).mkdir(exist_ok=True)
        
        tree_models = ['Random Forest', 'Gradient Boosting']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Importance', fontsize=16)
        
        for idx, model_name in enumerate(tree_models):
            if model_name in self.trained_models:
                model = self.trained_models[model_name]
                importances = model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1][:15]  # Top 15 features
                
                axes[idx].barh(range(len(indices)), importances[indices])
                axes[idx].set_yticks(range(len(indices)))
                axes[idx].set_yticklabels([feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name}')
                axes[idx].invert_yaxis()
                axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}/feature_importance.png")
        plt.close()
    
    def get_best_model(self):
        """Return the name and metrics of the best performing model"""
        if not self.results:
            print("No results available. Train models first.")
            return None
        
        # Best model based on test R² score
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\nBest Model: {best_model[0]}")
        print(f"Test R² Score: {best_model[1]['test_r2']:.4f}")
        print(f"Test RMSE: {best_model[1]['test_rmse']:.2f}")
        
        return best_model


if __name__ == "__main__":
    # Example usage
    from feature_engineering import FeatureEngineer
    
    # Prepare data
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.prepare_features_for_modeling()
    
    # Initialize predictor
    predictor = BikeRentalPredictor()
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train all models
    predictor.train_all_models(X_train, X_test, y_train, y_test)
    
    # Generate plots
    predictor.plot_model_comparison()
    predictor.plot_predictions_vs_actual(X_test, y_test)
    predictor.get_feature_importance(feature_names)
    
    # Get best model
    predictor.get_best_model()
