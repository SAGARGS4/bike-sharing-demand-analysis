# Bike Sharing Demand Analysis and Prediction

A comprehensive Python project for analyzing and predicting daily bike rental demand using machine learning techniques.

## 📊 Project Overview

This project analyzes the UCI Bike Sharing Dataset to understand bike rental patterns and build predictive models. It includes:
- **Exploratory Data Analysis (EDA)** with visualizations
- **Feature Engineering** to create meaningful predictors
- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting, and SVR
- **Model Evaluation** and comparison

## 🛠️ Tech Stack

- **Python 3.7+**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn

## 📁 Project Structure

```
bike-sharing-demand-analysis/
│
├── data/
│   ├── README.md           # Dataset information and download instructions
│   └── day.csv            # Dataset (to be downloaded)
│
├── plots/                 # Generated visualizations (created during runtime)
│
├── data_analysis.py       # Exploratory data analysis module
├── feature_engineering.py # Feature engineering and preprocessing
├── models.py             # Machine learning models
├── main.py               # Main pipeline orchestrator
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SAGARGS4/bike-sharing-demand-analysis.git
   cd bike-sharing-demand-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Visit [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
   - Download `day.csv`
   - Place it in the `data/` directory

### Usage

#### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Perform exploratory data analysis
2. Engineer features
3. Train all models (Linear Regression, Random Forest, Gradient Boosting, SVR)
4. Generate visualizations and evaluation metrics

#### Command Line Options

```bash
# Specify custom data path
python main.py --data path/to/day.csv

# Specify custom plots directory
python main.py --plots custom_plots

# Skip exploratory data analysis
python main.py --skip-analysis
```

#### Run Individual Modules

**Data Analysis Only:**
```python
from data_analysis import BikeDataAnalyzer

analyzer = BikeDataAnalyzer('data/day.csv')
analyzer.load_data()
analyzer.get_basic_info()
analyzer.generate_all_plots()
```

**Feature Engineering Only:**
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer('data/day.csv')
X, y, feature_names = engineer.prepare_features_for_modeling()
```

**Model Training Only:**
```python
from models import BikeRentalPredictor

predictor = BikeRentalPredictor()
X_train, X_test, y_train, y_test = predictor.split_data(X, y)
predictor.train_all_models(X_train, X_test, y_train, y_test)
```

## 📈 Features

### Data Analysis (`data_analysis.py`)
- **Temporal Patterns**: Daily, monthly, seasonal, and weekday trends
- **Weather Impact**: Temperature, humidity, wind speed correlations
- **User Analysis**: Casual vs registered user patterns
- **Correlation Matrix**: Feature relationships

### Feature Engineering (`feature_engineering.py`)
- Date component extraction (year, month, day)
- Interaction features (temp × humidity, temp × windspeed)
- Weather quality score
- Good biking day indicator
- Season indicators (one-hot encoding)
- Weekend indicator

### Machine Learning Models (`models.py`)

Four different regression models:

1. **Linear Regression**: Baseline model
2. **Random Forest**: Ensemble of decision trees
3. **Gradient Boosting**: Sequential ensemble method
4. **Support Vector Regression (SVR)**: Kernel-based approach

**Evaluation Metrics:**
- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

## 📊 Output Visualizations

The project generates several visualizations in the `plots/` directory:

1. **temporal_patterns.png**: Daily trends, monthly, seasonal, and weekday patterns
2. **weather_impact.png**: Weather conditions vs bike rentals
3. **correlation_matrix.png**: Feature correlation heatmap
4. **user_type_analysis.png**: Casual vs registered users
5. **model_comparison.png**: Performance comparison of all models
6. **predictions_vs_actual.png**: Predicted vs actual values for each model
7. **feature_importance.png**: Important features from tree-based models

## 📝 Dataset Information

The **UCI Bike Sharing Dataset** contains daily bike rental counts along with:
- **Temporal features**: Date, season, month, weekday, year
- **Weather features**: Temperature, humidity, wind speed, weather situation
- **User types**: Casual and registered users
- **Target**: Total rental count

For detailed dataset information, see `data/README.md`.

## 🎯 Results

The models are evaluated on:
- **R² Score**: Measures how well the model explains variance in the data
- **RMSE**: Average prediction error (lower is better)
- **MAE**: Average absolute prediction error (lower is better)

The best performing model is automatically identified and reported at the end of the pipeline.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Analyzing! 🚴‍♂️📊**