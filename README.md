# Demand Prediction Using LSTM

This project focuses on predicting demand using historical data and Long Short-Term Memory (LSTM) neural networks. The dataset used in this project is related to the **London Bike Sharing** system, and the goal is to forecast future demand based on historical time series data.

---

## Project Overview
Demand forecasting is essential for businesses to manage inventory, optimize supply chains, and make informed decisions. In this project, we leverage LSTM neural networks, a type of recurrent neural network (RNN), to capture temporal dependencies in time series data and predict future demand.

---

## Objectives
- Preprocess the time series data and prepare it for modeling.
- Build an LSTM model to predict future demand based on historical patterns.
- Evaluate model performance and visualize the forecasted results.

---

## Technologies Used
- **Python**: For programming and implementing the deep learning model.
- **TensorFlow/Keras**: Used to build and train the LSTM model.
- **Pandas and NumPy**: For data loading, preprocessing, and manipulation.
- **Matplotlib and Seaborn**: For data visualization.
- **Scikit-learn**: For splitting the dataset and evaluating model performance.

---

## Dataset
The dataset used in this project is the **London Bike Sharing** dataset, which contains the following features:
- **Timestamp**: The time at which the demand was recorded.
- **Demand**: The number of bikes rented at that timestamp.

The dataset was downloaded directly using `gdown`, and it is processed for further analysis and model training.

---

## Key Steps

1. **Data Preprocessing**:
   - The data is cleaned and transformed to handle missing values, if any, and the features are normalized for input into the LSTM model.
   - A sliding window is applied to generate the sequences of past demand to predict future demand.

2. **Modeling**:
   - The LSTM network is implemented using TensorFlow/Keras to capture temporal dependencies in the data.
   - The model's architecture includes an LSTM layer followed by Dense layers to output the forecasted demand.
   
3. **Evaluation**:
   - The model is evaluated using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
   - Visualization of the actual vs. predicted demand helps analyze the model's performance.

---

## How to Use

### Prerequisites
To run this project, ensure you have Python and the following libraries installed:

```bash
pip install tensorflow-gpu pandas numpy matplotlib scikit-learn seaborn gdown
