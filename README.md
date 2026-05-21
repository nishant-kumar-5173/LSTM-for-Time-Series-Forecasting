# README: Lab Experiment: LSTM for Time-Series Forecasting - Airline Passengers Dataset

## Overview
This repository contains a lab experiment focused on applying Long Short-Term Memory (LSTM) neural networks for time-series forecasting. The goal is to predict future international airline passenger counts using the classic Airline Passengers dataset, demonstrating a complete machine learning pipeline from data understanding to model evaluation and hyperparameter tuning.

## Dataset
*   **Name**: Airline Passengers Dataset
*   **Source**: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv`
*   **Description**: Monthly total of international airline passengers (in thousands).
*   **Type**: Univariate time series.
*   **Time Span**: January 1949 to December 1960 (144 monthly observations).
*   **Characteristics**: Exhibits a strong upward trend and clear annual seasonality with increasing variability over time (multiplicative seasonality).

## Methodology
The experiment follows a structured approach:

1.  **Data Access and Understanding**: Loaded the dataset, parsed dates, and inspected its structure and basic statistics.
2.  **Exploratory Analysis**: Visualized the time series to identify trend, seasonality, and changing variability. Confirmed an upward trend, strong annual seasonality (summer peaks), and increasing variability.
3.  **Time-Series Splitting**: Performed an 80/20 chronological train/test split to preserve temporal dependencies and avoid data leakage. The training period is Jan 1949 – Jul 1958, and the test period is Aug 1958 – Dec 1960.
4.  **Data Preprocessing**: Applied `MinMaxScaler` to normalize data between 0 and 1. Crucially, the scaler was fitted *only* on the training data to prevent data leakage from the test set. Sequences were then created using a sliding window approach, reshaping the data into the `(samples, timesteps, features)` format required by LSTMs.
5.  **Baseline Models**: Established benchmarks using a simple Linear Regression model (with lag features) and a Simple RNN, both with a lookback of 12.
6.  **LSTM Model Design and Training**: Developed a single-layer LSTM model using Keras/TensorFlow. The model was trained with the Adam optimizer and Mean Squared Error (MSE) loss, incorporating early stopping to prevent overfitting.
7.  **Prediction and Postprocessing**: Used the trained LSTM to make predictions on the test set, inverse-transformed the predictions back to the original scale, and visualized them against actual values.
8.  **Performance Evaluation**: Evaluated the LSTM and baseline models using standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
9.  **Hyperparameter Study**: Conducted experiments with different lookback window sizes, LSTM units, number of layers, and learning rates to assess their impact on model performance.
10. **Tricky Error Analysis**: Discussed common pitfalls in time-series forecasting, such as shuffling data, data leakage, and why simpler models can sometimes outperform LSTMs.
11. **Brainstorming and Improvement**: Explored more complex LSTM architectures like Stacked LSTM and Bidirectional LSTM, evaluating their potential benefits and challenges for this dataset.

## Key Findings
*   **LSTM Suitability**: LSTM proved effective for this dataset, successfully capturing both the strong upward trend and annual seasonality. It consistently outperformed the Linear Regression and Simple RNN baselines.
*   **Lookback Window**: A lookback window of 12 months (matching the seasonal cycle) was identified as optimal, providing the model with sufficient context to learn seasonal patterns.
*   **Data Integrity**: Strict adherence to chronological splitting and training-only scaling was crucial to avoid data leakage and ensure realistic model evaluation.
*   **Hyperparameter Sensitivity**: Model performance was particularly sensitive to the lookback window and learning rate.
*   **Limitations**: While the LSTM captured overall patterns well, it tended to produce slightly smoothed predictions and sometimes underestimated the amplitude of seasonal peaks, a common characteristic of MSE-trained models on smaller datasets.
*   **Further Improvements**: More complex architectures (Stacked/Bidirectional LSTM) showed marginal gains, suggesting that dataset size plays a significant role in leveraging such advanced models effectively.

## Usage
To run this experiment:

1.  Clone this repository.
2.  Install the required Python packages (e.g., `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`).
3.  Open and execute the Jupyter Notebook (`lab_experiment_lstm_time_series_forecasting.ipynb`) cell by cell.

The notebook is self-contained and provides detailed explanations for each step.
