"""
Main Script for Bike Sharing Demand Analysis and Prediction
Orchestrates the complete pipeline: data analysis, feature engineering, and modeling
"""

import argparse
import sys
from pathlib import Path

from data_analysis import BikeDataAnalyzer
from feature_engineering import FeatureEngineer
from models import BikeRentalPredictor


def run_analysis(data_path='data/day.csv', plots_dir='plots'):
    """
    Run exploratory data analysis
    
    Args:
        data_path: Path to the dataset
        plots_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    analyzer = BikeDataAnalyzer(data_path)
    
    try:
        analyzer.load_data()
        analyzer.get_basic_info()
        analyzer.generate_all_plots(plots_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the dataset following these steps:")
        print("1. Visit: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset")
        print("2. Download 'day.csv'")
        print("3. Place it in the 'data/' directory")
        sys.exit(1)


def run_feature_engineering(data_path='data/day.csv'):
    """
    Run feature engineering
    
    Args:
        data_path: Path to the dataset
    
    Returns:
        X, y, feature_names: Prepared features and target
    """
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    engineer = FeatureEngineer(data_path)
    X, y, feature_names = engineer.prepare_features_for_modeling()
    
    return X, y, feature_names


def run_modeling(X, y, feature_names, plots_dir='plots'):
    """
    Run machine learning modeling
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        plots_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("STEP 3: MACHINE LEARNING MODELING")
    print("="*70)
    
    predictor = BikeRentalPredictor()
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train all models
    predictor.train_all_models(X_train, X_test, y_train, y_test)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION AND VISUALIZATION")
    print("="*70)
    
    predictor.plot_model_comparison(plots_dir)
    predictor.plot_predictions_vs_actual(X_test, y_test, plots_dir)
    predictor.get_feature_importance(feature_names, plots_dir)
    
    # Get best model
    best_model = predictor.get_best_model()
    
    return predictor, best_model


def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(
        description='Bike Sharing Demand Analysis and Prediction'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/day.csv',
        help='Path to the dataset (default: data/day.csv)'
    )
    parser.add_argument(
        '--plots', 
        type=str, 
        default='plots',
        help='Directory to save plots (default: plots)'
    )
    parser.add_argument(
        '--skip-analysis', 
        action='store_true',
        help='Skip exploratory data analysis'
    )
    
    args = parser.parse_args()
    
    # Create plots directory
    Path(args.plots).mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("BIKE SHARING DEMAND ANALYSIS AND PREDICTION")
    print("="*70)
    print(f"\nDataset: {args.data}")
    print(f"Plots directory: {args.plots}")
    
    # Step 1: Data Analysis (optional)
    if not args.skip_analysis:
        run_analysis(args.data, args.plots)
    
    # Step 2: Feature Engineering
    X, y, feature_names = run_feature_engineering(args.data)
    
    # Step 3 & 4: Modeling and Evaluation
    predictor, best_model = run_modeling(X, y, feature_names, args.plots)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal samples: {len(X)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"\nBest performing model: {best_model[0]}")
    print(f"  - Test R² Score: {best_model[1]['test_r2']:.4f}")
    print(f"  - Test RMSE: {best_model[1]['test_rmse']:.2f}")
    print(f"  - Test MAE: {best_model[1]['test_mae']:.2f}")
    
    print(f"\nAll plots saved in '{args.plots}/' directory")
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
