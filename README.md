# Stock Market ML Pipeline: Classification & Regression Analysis

## Overview
This project implements a comprehensive Machine Learning pipeline for stock market analysis, encompassing both **classification** (predicting stock movement: Up/Down) and **regression** (predicting actual Close price). It leverages a variety of individual and ensemble learning models, extensive feature engineering, and detailed evaluation metrics, culminating in insightful visualizations and a summary report.

## Features
- **Data Loading & EDA**: Initial data loading and exploratory data analysis to understand dataset characteristics.
- **Feature Engineering**: Creation of new technical indicator features to enhance model performance.
- **Data Splitting**: Robust data splitting for both classification (stratified) and regression tasks.
- **Classification Models**: Training and evaluation of multiple classification algorithms, including Decision Trees, Random Forests, Gradient Boosting, XGBoost, LightGBM, and various ensemble methods (Voting, Stacking).
- **Regression Models**: Training and evaluation of diverse regression algorithms, including Linear Models, Decision Trees, Random Forests, Gradient Boosting, XGBoost, LightGBM, and ensemble methods.
- **Hyperparameter Tuning**: Example of GridSearchCV for optimizing model parameters.
- **Comprehensive Evaluation**: Calculation of a wide array of metrics for both classification (Accuracy, Precision, Recall, F1-Score, AUC-ROC, MCC, Kappa, Log Loss) and regression (MAE, MSE, RMSE, MAPE, R², Explained Variance).
- **Visualizations**: Generation of 12 distinct plots to visualize model performance, feature importance, and predictions.
- **Summary Report**: A final, concise report summarizing the best-performing models and key outcomes.

## Installation
To set up the project locally, follow these steps:

1.  **Clone the repository** (assuming this code is part of a repository):
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    The project relies on common Python data science libraries. You can install them using pip:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
    ```

## Usage

1.  **Prepare your data**: Ensure you have a CSV file named `ml_dataset1.csv` in the expected directory (`/mnt/user-data/uploads/` as per the script, but you might need to adjust this path or place the file in the same directory as the script for local execution). This dataset should contain relevant stock data with a 'Target' column for classification and a 'Close' column for regression.

2.  **Run the script**:
    ```bash
    python stock_ml_pipeline.py
    ```

3.  **Outputs**: The script will print detailed evaluation metrics to the console and save several visualization PNG files and prediction CSVs to the `/mnt/user-data/outputs/` directory (again, adjust this path as needed).

    -   `classification_predictions.csv`
    -   `regression_predictions.csv`
    -   `fig1_feature_correlation.png`
    -   `fig2_target_distribution.png`
    -   `fig3_classification_comparison.png`
    -   `fig4_regression_comparison.png`
    -   `fig5_confusion_matrices.png`
    -   `fig6_roc_curves.png`
    -   `fig7_actual_vs_predicted.png`
    -   `fig8_residual_analysis.png`
    -   `fig9_feature_importance.png`
    -   `fig10_precision_recall.png`
    -   `fig11_cv_boxplots.png`
    -   `fig12_price_predictions.png`

## Data
The script expects a CSV file named `ml_dataset1.csv`. This file should contain historical stock data. The `Target` column is used for classification (e.g., 0 for 'Down', 1 for 'Up'), and the `Close` price is used for regression.

## Models & Algorithms

### Classification
-   **Individual Models**: Decision Tree
-   **Bagging Family**: Random Forest, Extra Trees, Bagging (with Decision Tree estimator)
-   **Boosting Family**: AdaBoost, Gradient Boosting, XGBoost, LightGBM
-   **Ensemble Methods**: Voting Classifier (Hard & Soft), Stacking Classifier

### Regression
-   **Individual Models**: Decision Tree
-   **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
-   **Bagging Family**: Random Forest, Extra Trees, Bagging (with Decision Tree estimator)
-   **Boosting Family**: AdaBoost, Gradient Boosting, XGBoost, LightGBM
-   **Ensemble Methods**: Voting Regressor, Stacking Regressor

## Evaluation Metrics

### Classification Metrics
-   Accuracy, Precision, Recall, F1-Score
-   AUC-ROC, Matthews Correlation Coefficient (MCC), Cohen's Kappa
-   Log Loss, Cross-Validation Accuracy

### Regression Metrics
-   Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
-   Mean Absolute Percentage Error (MAPE)
-   R² Score, Explained Variance Score, Cross-Validation R²

## Visualizations
The pipeline generates a series of plots to provide a deep understanding of the data and model performance:

1.  **Feature Correlation Heatmap**: Visualizes the correlation between features.
2.  **Target Distribution**: Shows the distribution of the classification target.
3.  **Classification Model Comparison**: Bar plots comparing various classification metrics across models.
4.  **Regression Model Comparison**: Bar plots comparing various regression metrics across models.
5.  **Confusion Matrices**: For the top 4 classification models.
6.  **ROC Curves**: For all classification models.
7.  **Actual vs Predicted**: Scatter plots for the top 4 regression models.
8.  **Residual Analysis**: Plots for the best regression model (residuals vs predicted, residual distribution, Q-Q plot).
9.  **Feature Importance**: Bar plots for classification (Random Forest Tuned) and regression (XGBoost Tuned).
10. **Precision-Recall Curves**: For all classification models.
11. **Cross-Validation Score Box-plots**: Distribution of CV scores for both classification and regression models.
12. **Predicted Stock Prices**: Time series plot comparing actual vs. predicted close prices for top 3 regressors.

## Results Summary
The script concludes with a summary report highlighting the top 3 performing models for both classification and regression based on key metrics (AUC-ROC for classification, R² for regression), along with details on hyperparameter tuning results.

## Dependencies
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `xgboost`
-   `lightgbm`

