"""
Example Usage Script
Demonstrates how to use the bike sharing analysis modules
"""

from data_analysis import BikeDataAnalyzer
from feature_engineering import FeatureEngineer
from models import BikeRentalPredictor


def example_1_data_analysis():
    """Example 1: Perform data analysis only"""
    print("\n" + "="*60)
    print("Example 1: Data Analysis")
    print("="*60)
    
    analyzer = BikeDataAnalyzer('data/day.csv')
    analyzer.load_data()
    analyzer.get_basic_info()
    analyzer.generate_all_plots('plots')


def example_2_feature_engineering():
    """Example 2: Feature engineering only"""
    print("\n" + "="*60)
    print("Example 2: Feature Engineering")
    print("="*60)
    
    engineer = FeatureEngineer('data/day.csv')
    X, y, feature_names = engineer.prepare_features_for_modeling()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures: {feature_names[:5]}... (showing first 5)")


def example_3_model_training():
    """Example 3: Train and evaluate models"""
    print("\n" + "="*60)
    print("Example 3: Model Training and Evaluation")
    print("="*60)
    
    # Prepare features
    engineer = FeatureEngineer('data/day.csv')
    X, y, feature_names = engineer.prepare_features_for_modeling()
    
    # Initialize predictor
    predictor = BikeRentalPredictor()
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train all models
    predictor.train_all_models(X_train, X_test, y_train, y_test)
    
    # Get best model
    best = predictor.get_best_model()
    
    # Generate visualizations
    predictor.plot_model_comparison('plots')
    predictor.plot_predictions_vs_actual(X_test, y_test, 'plots')
    predictor.get_feature_importance(feature_names, 'plots')


def example_4_train_single_model():
    """Example 4: Train a single specific model"""
    print("\n" + "="*60)
    print("Example 4: Train Single Model (Random Forest)")
    print("="*60)
    
    # Prepare features
    engineer = FeatureEngineer('data/day.csv')
    X, y, feature_names = engineer.prepare_features_for_modeling()
    
    # Initialize predictor
    predictor = BikeRentalPredictor()
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train only Random Forest
    model = predictor.train_model('Random Forest', X_train, y_train)
    
    # Evaluate
    metrics = predictor.evaluate_model('Random Forest', model, 
                                      X_train, X_test, y_train, y_test)
    
    print(f"\nModel trained and evaluated!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        if example_num == '1':
            example_1_data_analysis()
        elif example_num == '2':
            example_2_feature_engineering()
        elif example_num == '3':
            example_3_model_training()
        elif example_num == '4':
            example_4_train_single_model()
        else:
            print("Invalid example number. Use 1, 2, 3, or 4.")
    else:
        print("Usage: python example_usage.py <example_number>")
        print("\nAvailable examples:")
        print("  1 - Data Analysis")
        print("  2 - Feature Engineering")
        print("  3 - Model Training and Evaluation")
        print("  4 - Train Single Model")
        print("\nExample: python example_usage.py 1")
