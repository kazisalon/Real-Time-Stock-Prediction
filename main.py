# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class StockPricePredictionModel:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize the stock prediction model
        
        :param ticker: Stock ticker symbol
        :param start_date: Start date for historical data
        :param end_date: End date for historical data
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        
    def fetch_stock_data(self):
        """
        Fetch historical stock data from Yahoo Finance
        """
        print(f"Fetching stock data for {self.ticker}")
        self.data = yf.download(self.ticker, 
                                start=self.start_date, 
                                end=self.end_date)
        return self.data
    
    def preprocess_data(self, look_back=60):
        """
        Preprocess stock data for LSTM model
        
        :param look_back: Number of previous time steps to use for prediction
        """
        # Select closing prices
        dataset = self.data['Close'].values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create training dataset
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return scaler
    
    def build_lstm_model(self, look_back=60):
        """
        Build LSTM neural network model
        
        :param look_back: Number of previous time steps to use for prediction
        """
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model
    
    def train_model(self, epochs=50, batch_size=32):
        """
        Train the LSTM model
        
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        """
        history = self.model.fit(
            self.X_train, self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(self.X_test, self.y_test),
            verbose=1
        )
        return history
    
    def predict_prices(self, scaler):
        """
        Predict future stock prices
        
        :param scaler: MinMaxScaler used for data normalization
        """
        # Make predictions
        predictions = self.model.predict(self.X_test)
        predictions = scaler.inverse_transform(predictions)
        
        return predictions
    
    def prophet_forecast(self):
        """
        Use Prophet for time series forecasting
        """
        prophet_data = self.data.reset_index()[['Date', 'Close']]
        prophet_data.columns = ['ds', 'y']
        
        model = Prophet(
            daily_seasonality=True, 
            yearly_seasonality=True
        )
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        return forecast
    
    def visualize_results(self, actual, predicted, forecast=None):
        """
        Visualize prediction results
        
        :param actual: Actual stock prices
        :param predicted: Predicted stock prices
        :param forecast: Optional Prophet forecast
        """
        plt.figure(figsize=(15,6))
        
        # Actual vs Predicted
        plt.subplot(1,2,1)
        plt.plot(actual, label='Actual Prices', color='blue')
        plt.plot(predicted, label='Predicted Prices', color='red')
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Prophet Forecast
        if forecast is not None:
            plt.subplot(1,2,2)
            plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
            plt.fill_between(forecast['ds'], 
                             forecast['yhat_lower'], 
                             forecast['yhat_upper'], 
                             alpha=0.3)
            plt.title(f'{self.ticker} Prophet Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
        
        plt.tight_layout()
        plt.show()

def main():
    # Parameters
    TICKER = 'AAPL'  # Example: Apple Inc.
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    
    # Initialize and run prediction model
    predictor = StockPricePredictionModel(TICKER, START_DATE, END_DATE)
    
    # Fetch and preprocess data
    data = predictor.fetch_stock_data()
    scaler = predictor.preprocess_data()
    
    # Build and train LSTM model
    model = predictor.build_lstm_model()
    history = predictor.train_model()
    
    # Predict prices
    predicted_prices = predictor.predict_prices(scaler)
    
    # Prophet forecast
    prophet_forecast = predictor.prophet_forecast()
    
    # Visualize results
    predictor.visualize_results(
        predictor.data['Close'].values[-len(predicted_prices):], 
        predicted_prices, 
        prophet_forecast
    )

if __name__ == "__main__":
    main()

# Installation Requirements
