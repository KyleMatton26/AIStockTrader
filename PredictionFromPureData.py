import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Download and prepare data
ticker = "AAPL"
stock_data = yf.download(ticker, start="2000-01-01", end="2024-08-04")
data = stock_data['Close'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Define parameters
past_days = 30
future_days = 3

# Create dataset
X = []
y = []

for i in range(past_days, len(data_normalized) - future_days + 1):
    X.append(data_normalized[i - past_days:i])
    y.append(data_normalized[i:i + future_days])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the LSTM model
class StockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=future_days):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize model, loss function, and optimizer
model = StockPredictor().to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Predict the stock price for the test set
model.eval()
with torch.no_grad():
    predictions_normalized = model(X_test).cpu().numpy()

# Inverse transform the normalized predictions
predictions = scaler.inverse_transform(predictions_normalized)

# Print the actual last 3 days before prediction for comparison
last_3_days_actual = data[-3:]
print(f'Actual last 3-day prices: {last_3_days_actual}')

print(f'Predicted next 3-day prices for test set: {predictions}')

# Predict the next 3 days based on the last 30 days of available data
last_30_days = data_normalized[-past_days:]
last_30_days = torch.tensor(last_30_days, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

model.eval()
with torch.no_grad():
    future_prediction_normalized = model(last_30_days.to('cuda' if torch.cuda.is_available() else 'cpu')).cpu().numpy()
    future_prediction = scaler.inverse_transform(future_prediction_normalized)

print(f'Predicted next 3-day prices based on the last 30 days: {future_prediction[0]}')
