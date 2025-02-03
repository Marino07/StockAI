import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gc
gc.enable()

# ...existing imports...
from datetime import datetime, timedelta
import plotly.graph_objects as go
import jinja2

try:
    print("Checking PyTorch installation...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except Exception as e:
    print(f"Error importing PyTorch: {e}")
    exit(1)

# ========== CONFIGURATION ========== 
STOCK_SYMBOLS = ["TSLA", "AAPL"]
COLORS = {'TSLA': '#e31937', 'AAPL': '#555555'}
PERIOD = "2y"            # Increased to 2 years for better historical context
SEQUENCE_LENGTH = 30     # Reduced to 30 days for shorter trend focus
EPOCHS = 10           # Increased to 50 for better learning
BATCH_SIZE = 32
LEARNING_RATE = 0.0001   # Reduced learning rate for more stable learning
FUTURE_DAYS = 5
HIDDEN_SIZE = 128        # Increased for better memory
NUM_LAYERS = 2
MAX_DAILY_CHANGE = 0.03  # Reduced maximum daily change to 3%
MIN_DAILY_CHANGE = -0.03 # Added minimum daily change

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== FETCH STOCK DATA ========== 
def fetch_stock_data(symbol, period="5y"):
    df = yf.download(symbol, period=period)
    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {symbol}")
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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, output_size)
        self.tanh = nn.Tanh()  # Limits output
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.tanh(out) * MAX_DAILY_CHANGE  # Limits change to MAX_DAILY_CHANGE
        return out

# ========== TRAIN THE MODEL ========== 
def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    best_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).squeeze()
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            if y_pred.shape != y_batch.shape:
                y_batch = y_batch.view_as(y_pred)
            
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Add gradient clipping for more stable learning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

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
    
    # Fix for the last known price
    last_known_price = df['Close'].iloc[-1]
    if isinstance(last_known_price, pd.Series):
        last_known_price = last_known_price.iloc[0]
    
    # Fix for historical volatility calculation
    historical_prices = df['Close'].values[-30:].flatten()  # Ensure 1D array
    
    # Calculate returns only if we have enough data
    if (len(historical_prices) > 1):
        daily_returns = np.diff(historical_prices) / historical_prices[:-1]
        avg_daily_change = np.mean(np.abs(daily_returns))
        volatility = np.std(daily_returns)
    else:
        avg_daily_change = MAX_DAILY_CHANGE / 2
        volatility = MAX_DAILY_CHANGE / 4
    
    # Extract and prepare the last sequence
    last_sequence = df.iloc[-sequence_length:][['Open', 'High', 'Low', 'Volume']].values
    last_sequence_scaled = scaler_X.transform(last_sequence)
    current_input = last_sequence_scaled.copy()

    for day in range(future_days):
        with torch.no_grad():
            current_tensor = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            predicted_change = model(current_tensor).cpu().numpy()[0][0]
            
            # Limit the change based on historical volatility
            max_allowed_change = min(MAX_DAILY_CHANGE, avg_daily_change + volatility)
            predicted_change = np.clip(predicted_change, -max_allowed_change, max_allowed_change)
            
            # Calculate predicted price with exponential dampening for later days
            if day == 0:
                predicted_price = last_known_price * (1 + predicted_change)
            else:
                dampening_factor = np.exp(-day * 0.3)
                change = predicted_change * dampening_factor
                predicted_price = future_predictions[-1] * (1 + change)
            
            # Additional constraints based on historical data
            historical_std = np.std(historical_prices)
            max_change = historical_std * (day + 1) * 0.5
            
            upper_bound = last_known_price * (1 + MAX_DAILY_CHANGE * (day + 1))
            lower_bound = last_known_price * (1 - MAX_DAILY_CHANGE * (day + 1))
            
            predicted_price = np.clip(predicted_price, lower_bound, upper_bound)
            future_predictions.append(float(predicted_price))
            
            # Update input for next prediction
            typical_spread = np.mean(df['High'].values - df['Low'].values) / df['Close'].values[-1]
            new_row = np.array([
                predicted_price * (1 - typical_spread/2),
                predicted_price * (1 + typical_spread/2),
                predicted_price * (1 - typical_spread/2),
                df['Volume'].mean()
            ]).reshape(1, -1)
            
            new_row_scaled = scaler_X.transform(new_row)
            current_input = np.append(current_input[1:], new_row_scaled, axis=0)

    return future_predictions

# Nova funkcija za naprednu vizualizaciju
def create_advanced_visualization(symbol, df, actual_prices, predictions, future_dates, future_pred):
    # Kreiramo glavni figure
    fig = go.Figure()
    
    # Dodajemo candlestick chart za povijesne podatke
    fig.add_trace(
        go.Candlestick(
            x=df.index[-90:],
            open=df['Open'][-90:],
            high=df['High'][-90:],
            low=df['Low'][-90:],
            close=df['Close'][-90:],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
    )
    
    # Dodajemo stvarne cijene kao liniju
    fig.add_trace(
        go.Scatter(
            x=actual_prices.index,
            y=actual_prices,
            name='Actual Price',
            line=dict(color='#2196f3', width=2)
        )
    )
    
    # Pripremamo podatke za predviđanja (spajamo zadnju stvarnu cijenu s predviđanjima)
    last_actual_date = actual_prices.index[-1]
    last_actual_price = actual_prices.iloc[-1]
    
    pred_dates = [last_actual_date] + list(future_dates)
    pred_values = [last_actual_price] + list(future_pred)
    
    # Dodajemo predviđanja kao isprekidanu liniju
    fig.add_trace(
        go.Scatter(
            x=pred_dates,
            y=pred_values,
            name='Predicted Price',
            line=dict(color='#ff9800', width=3, dash='dash'),
            mode='lines+markers'
        )
    )
    
    # Poboljšani layout
    fig.update_layout(
        title={
            'text': f"{symbol} Stock Analysis & Prediction",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#1a237e')
        },
        yaxis=dict(
            title="Price ($)",
            titlefont=dict(size=16),
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(70, 70, 70, 0.1)',
            zerolinecolor='rgba(70, 70, 70, 0.1)'
        ),
        xaxis=dict(
            title="Date",
            titlefont=dict(size=16),
            tickfont=dict(size=14),
            rangeslider=dict(visible=True),
            showgrid=True,
            gridcolor='rgba(70, 70, 70, 0.1)',
            zerolinecolor='rgba(70, 70, 70, 0.1)'
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(70, 70, 70, 0.2)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        height=800,
        width=1400,
        margin=dict(l=60, r=60, t=80, b=50)
    )
    
    # Dodaj oznake za posljednju stvarnu cijenu i predviđanje
    fig.add_annotation(
        x=last_actual_date,
        y=last_actual_price,
        text=f"Last: ${last_actual_price:.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=future_dates[-1],
        y=future_pred[-1],
        text=f"Prediction: ${future_pred[-1]:.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    return fig

def save_visualization(figures, symbols):
    template_loader = jinja2.FileSystemLoader(searchpath="./templates/")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("analysis_template.html")
    
    os.makedirs("output", exist_ok=True)
    
    for fig, symbol in zip(figures, symbols):
        output_file = f"output/{symbol}_analysis.html"
        
        # Ažuriramo layout prije generiranja HTML-a
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(240,240,240,0.8)',
            width=1200,
            height=800
        )
        
        # Generiramo HTML s plotly grafom
        plot_div = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={'displayModeBar': True}
        )
        
        # Renderiramo template s grafom
        html_content = template.render(
            title=f"{symbol} Stock Analysis",
            plot_div=plot_div
        )
        
        # Spremamo HTML
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Analysis saved to {output_file}")

# ...existing code...

if __name__ == "__main__":
    try:
        predictions_dict = {}
        actual_dict = {}
        figures = []
        
        for symbol in STOCK_SYMBOLS:
            print(f"\nProcessing {symbol}...")
            
            # Fetch with longer period
            df = fetch_stock_data(symbol, period=PERIOD)
            X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(df)
            
            # Initialize and train model
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            model = StockLSTM(
                input_size=4, 
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                output_size=1
            ).to(DEVICE)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            
            print(f"Training model for {symbol}...")
            train_model(model, train_loader, criterion, optimizer, EPOCHS)
            
            # Make predictions
            actual_prices = pd.Series(
                df['Close'].values[-90:].flatten(),
                index=df.index[-90:]
            )
            
            future_predictions = predict_future(model, df, SEQUENCE_LENGTH, FUTURE_DAYS, scaler_X, scaler_y)
            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1), 
                periods=FUTURE_DAYS,
                freq='B'
            )
            
            # Convert predictions to list
            future_predictions = np.array(future_predictions).flatten().tolist()
            
            predictions_dict[symbol] = {
                'predictions': scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                'future': future_predictions
            }
            actual_dict[symbol] = actual_prices
            
            # Create visualization
            fig = create_advanced_visualization(
                symbol,
                df,
                actual_prices,
                predictions_dict[symbol]['predictions'],
                future_dates,
                future_predictions
            )
            figures.append(fig)
            
        # Save visualizations
        save_visualization(figures, STOCK_SYMBOLS)
        
        print("\nAnalyses have been saved to the output directory.")
        print("Open the HTML files in your web browser to view the results.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        gc.collect()
