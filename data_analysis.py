"""
Bike Sharing Data Analysis Module
Performs exploratory data analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class BikeDataAnalyzer:
    """Class for analyzing bike sharing data"""
    
    def __init__(self, data_path='data/day.csv'):
        """Initialize analyzer with data path"""
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        """Load the bike sharing dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download day.csv from UCI Bike Sharing Dataset and place it in the data/ directory."
            )
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully with {len(self.df)} records")
        return self.df
    
    def get_basic_info(self):
        """Print basic information about the dataset"""
        if self.df is None:
            self.load_data()
        
        print("\n=== Dataset Information ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        
    def plot_temporal_patterns(self, save_path='plots'):
        """Create visualizations for temporal patterns"""
        if self.df is None:
            self.load_data()
        
        # Create plots directory if it doesn't exist
        Path(save_path).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Patterns in Bike Rentals', fontsize=16, y=1.00)
        
        # Daily trend
        axes[0, 0].plot(self.df.index, self.df['cnt'], alpha=0.6)
        axes[0, 0].set_title('Daily Bike Rentals Over Time')
        axes[0, 0].set_xlabel('Day Index')
        axes[0, 0].set_ylabel('Total Rentals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Monthly average
        monthly_avg = self.df.groupby('mnth')['cnt'].mean()
        axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='skyblue')
        axes[0, 1].set_title('Average Rentals by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Rentals')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Seasonal pattern
        seasonal_avg = self.df.groupby('season')['cnt'].mean()
        season_labels = ['Spring', 'Summer', 'Fall', 'Winter']
        axes[1, 0].bar(range(1, 5), seasonal_avg.values, color='lightgreen')
        axes[1, 0].set_title('Average Rentals by Season')
        axes[1, 0].set_xlabel('Season')
        axes[1, 0].set_ylabel('Average Rentals')
        axes[1, 0].set_xticks(range(1, 5))
        axes[1, 0].set_xticklabels(season_labels)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Weekday pattern
        weekday_avg = self.df.groupby('weekday')['cnt'].mean()
        weekday_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        axes[1, 1].bar(range(7), weekday_avg.values, color='coral')
        axes[1, 1].set_title('Average Rentals by Weekday')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Rentals')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(weekday_labels)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/temporal_patterns.png', dpi=300, bbox_inches='tight')
        print(f"Temporal patterns plot saved to {save_path}/temporal_patterns.png")
        plt.close()
        
    def plot_weather_impact(self, save_path='plots'):
        """Create visualizations for weather impact"""
        if self.df is None:
            self.load_data()
        
        Path(save_path).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Weather Impact on Bike Rentals', fontsize=16, y=1.00)
        
        # Temperature vs rentals
        axes[0, 0].scatter(self.df['temp'], self.df['cnt'], alpha=0.5, color='orange')
        axes[0, 0].set_title('Temperature vs Rentals')
        axes[0, 0].set_xlabel('Normalized Temperature')
        axes[0, 0].set_ylabel('Total Rentals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Humidity vs rentals
        axes[0, 1].scatter(self.df['hum'], self.df['cnt'], alpha=0.5, color='blue')
        axes[0, 1].set_title('Humidity vs Rentals')
        axes[0, 1].set_xlabel('Normalized Humidity')
        axes[0, 1].set_ylabel('Total Rentals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Windspeed vs rentals
        axes[1, 0].scatter(self.df['windspeed'], self.df['cnt'], alpha=0.5, color='green')
        axes[1, 0].set_title('Wind Speed vs Rentals')
        axes[1, 0].set_xlabel('Normalized Wind Speed')
        axes[1, 0].set_ylabel('Total Rentals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Weather situation
        weather_avg = self.df.groupby('weathersit')['cnt'].mean()
        axes[1, 1].bar(weather_avg.index, weather_avg.values, color='purple')
        axes[1, 1].set_title('Average Rentals by Weather Situation')
        axes[1, 1].set_xlabel('Weather Situation')
        axes[1, 1].set_ylabel('Average Rentals')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/weather_impact.png', dpi=300, bbox_inches='tight')
        print(f"Weather impact plot saved to {save_path}/weather_impact.png")
        plt.close()
        
    def plot_correlation_matrix(self, save_path='plots'):
        """Create correlation matrix heatmap"""
        if self.df is None:
            self.load_data()
        
        Path(save_path).mkdir(exist_ok=True)
        
        # Select relevant numeric columns
        numeric_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                       'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 
                       'casual', 'registered', 'cnt']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_path}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {save_path}/correlation_matrix.png")
        plt.close()
        
    def plot_user_type_analysis(self, save_path='plots'):
        """Analyze casual vs registered users"""
        if self.df is None:
            self.load_data()
        
        Path(save_path).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('User Type Analysis', fontsize=16)
        
        # Casual vs Registered over time
        axes[0].plot(self.df.index, self.df['casual'], alpha=0.6, label='Casual', color='blue')
        axes[0].plot(self.df.index, self.df['registered'], alpha=0.6, label='Registered', color='red')
        axes[0].set_title('Casual vs Registered Users Over Time')
        axes[0].set_xlabel('Day Index')
        axes[0].set_ylabel('Number of Users')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Proportion pie chart
        total_casual = self.df['casual'].sum()
        total_registered = self.df['registered'].sum()
        axes[1].pie([total_casual, total_registered], 
                   labels=['Casual', 'Registered'],
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightcoral'],
                   startangle=90)
        axes[1].set_title('Overall User Type Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/user_type_analysis.png', dpi=300, bbox_inches='tight')
        print(f"User type analysis saved to {save_path}/user_type_analysis.png")
        plt.close()
        
    def generate_all_plots(self, save_path='plots'):
        """Generate all visualization plots"""
        print("\nGenerating visualizations...")
        self.plot_temporal_patterns(save_path)
        self.plot_weather_impact(save_path)
        self.plot_correlation_matrix(save_path)
        self.plot_user_type_analysis(save_path)
        print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    # Example usage
    analyzer = BikeDataAnalyzer()
    analyzer.load_data()
    analyzer.get_basic_info()
    analyzer.generate_all_plots()
