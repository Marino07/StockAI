import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ========== CONFIGURATION ========== 
STOCK_SYMBOL = "NVDA"
SEQUENCE_LENGTH = 60  
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0009  
FUTURE_DAYS = 5  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== FETCH STOCK DATA ========== 
def fetch_stock_data(symbol, period="5y"):
    df = yf.download(symbol, period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  
    df.dropna(inplace=True)
    return df

# ========== PREPROCESS DATA ========== 
def preprocess_data(df, sequence_length=SEQUENCE_LENGTH, test_size=0.2):
    # Separate features and target
    features = df[['Open', 'High', 'Low', 'Volume']]  # Exclude 'Close' here
    target = df[['Close']]  # 'Close' is the target
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(features)  
    y_scaled = scaler_y.fit_transform(target)  

    X, y = [], []
    for i in range(len(X_scaled) - sequence_length):
        X.append(X_scaled[i:i+sequence_length])
        y.append(y_scaled[i+sequence_length, 0])  

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y  

# ========== DEFINE LSTM MODEL ========== 
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  

# ========== TRAIN THE MODEL ========== 
def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).squeeze()  # Ensure y_batch has the correct shape

            optimizer.zero_grad()
            y_pred = model(X_batch)

            # Ensure y_pred and y_batch have the same shape
            if y_pred.shape != y_batch.shape:
                y_batch = y_batch.view_as(y_pred)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")


# ========== PREDICT ========== 
def predict(model, X_test, scaler_y):
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_pred_scaled = model(X_test).cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)  
    return y_pred

# ========== PREDICT FUTURE PRICES (FROM ACTUAL LINE) ========== 
def predict_future(model, df, sequence_length, future_days, scaler_X, scaler_y):
    model.eval()
    future_predictions = []
    
    # Extract the last `sequence_length` rows for prediction (without Volume)
    last_sequence = df.iloc[-sequence_length:][['Open', 'High', 'Low', 'Close']].values
    last_sequence_scaled = scaler_X.transform(last_sequence)  # Scale the sequence

    current_input = last_sequence_scaled.copy()

    for _ in range(future_days):
        with torch.no_grad():
            current_input_tensor = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Predict the next day's Close price (scaled)
            next_day_pred_scaled = model(current_input_tensor).cpu().numpy()
            next_day_pred = scaler_y.inverse_transform(next_day_pred_scaled.reshape(-1, 1))  # Inverse transform

            future_predictions.append(next_day_pred[0, 0])  

            # Now prepare for the next prediction:
            # We keep the predicted Close price and estimate Open, High, Low based on ratios
            last_real = current_input[-1]
            open_ratio = last_real[0] / last_real[3]  # Open/Close ratio
            high_ratio = last_real[1] / last_real[3]  # High/Close ratio
            low_ratio = last_real[2] / last_real[3]   # Low/Close ratio

            # Create new predicted values based on the ratios and predicted Close
            estimated_open = next_day_pred[0, 0] * open_ratio
            estimated_high = next_day_pred[0, 0] * high_ratio
            estimated_low = next_day_pred[0, 0] * low_ratio

            # Now create the new input for the model with the predicted values
            new_input = np.array([estimated_open, estimated_high, estimated_low, next_day_pred[0, 0]]).reshape(1, -1)

            # Update the input for the next step (slide the window)
            current_input = np.append(current_input[1:], new_input, axis=0)  # Slide the window

    return future_predictions

# ========== MAIN SCRIPT ========== 
if __name__ == "__main__":
    print("Fetching stock data...")
    df = fetch_stock_data(STOCK_SYMBOL)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(df)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = StockLSTM(input_size=4, hidden_size=64, num_layers=1, output_size=1).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    print("Making predictions...")
    y_pred = predict(model, X_test, scaler_y)

    print("Making future predictions...")
    predicted_future_prices = predict_future(model, df, SEQUENCE_LENGTH, FUTURE_DAYS, scaler_X, scaler_y)

    # ========== PLOTTING ========== 
actual_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))  

plt.figure(figsize=(14, 7))

# Plot actual prices (blue line)
plt.plot(df.index[-len(y_test):], actual_prices, label="Actual Prices", color="blue")

# Plot future predictions (green line, continuing from the last actual price)
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS)
last_actual_price = actual_prices[-1]  # Get the last actual price from the test set

# The predicted future prices should start from the last actual price.
predicted_future_prices = np.insert(predicted_future_prices, 0, last_actual_price)  

plt.plot(np.append(df.index[-1:], future_dates), predicted_future_prices, label="Future Predictions", color="green", linestyle="dashed")

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{STOCK_SYMBOL} Stock Price Prediction with Future Forecast")
plt.legend()
plt.show()
