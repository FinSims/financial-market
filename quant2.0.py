from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

# Load AMZN stock data
msft = yf.Ticker("UNH")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get historical market data
df = msft.history(period="5y")

# Focus on the 'Close' column
df = df[['Close']]

# Plot the closing price over time
plt.plot(df.index, df['Close'])
plt.show()

# Function to prepare dataframe for LSTM model


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df


# Prepare the data with a lookback period
lookback = 7
shifted_df = prepare_dataframe_for_lstm(df, lookback)

# Convert the dataframe to numpy array
shifted_df_np = shifted_df.to_numpy()

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_np = scaler.fit_transform(shifted_df_np)

# Split into input (X) and output (y)
X = shifted_df_np[:, 1:]
y = shifted_df_np[:, 0]

# Reverse the input array for the LSTM
X = dc(np.flip(X, axis=1))

# Split into training and testing sets
split_index = int(len(X) * 0.95)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Reshape the data for LSTM input
X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Convert to PyTorch tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Define the custom Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM model


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size,
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size,
                         self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Initialize model, loss function, and optimizer
model = LSTM(1, 16, 2).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function


def train_one_epoch():
    model.train(True)
    running_loss = 0.0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(
                batch_index + 1, avg_loss_across_batches))
            running_loss = 0.0
    print()

# Validation function


def validate_one_epoch():
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    avg_loss_across_batches = running_loss / len(test_loader)
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


# Train the model
num_epochs = 30
for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

# Plot training predictions vs actual values
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()
plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# Inverse transform the predictions and actual values for plotting
train_predictions = predicted.flatten()
dummies = np.zeros((X_train.shape[0], lookback + 1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)
train_predictions = dc(dummies[:, 0])

dummies = np.zeros((X_train.shape[0], lookback + 1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)
new_y_train = dc(dummies[:, 0])

plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# Plot test predictions vs actual values
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
dummies = np.zeros((X_test.shape[0], lookback + 1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)
test_predictions = dc(dummies[:, 0])

dummies = np.zeros((X_test.shape[0], lookback + 1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)
new_y_test = dc(dummies[:, 0])

plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# Prepare the results dataframe
test_dates = df.index[-len(new_y_test):]
test_results_df = pd.DataFrame({
    'Date': test_dates,
    'Actual': new_y_test,
    'Predicted': test_predictions
})
print("Test Results:")
print(test_results_df)

# Function to predict future dates


def predict_future_dates(model, last_known_data, scaler, num_predictions):
    model.eval()
    predictions = []
    data = last_known_data
    for _ in range(num_predictions):
        input_data = torch.tensor(data.reshape(
            1, lookback, 1)).float().to(device)
        with torch.no_grad():
            pred = model(input_data).cpu().numpy().flatten()
        predictions.append(pred[0])
        data = np.append(data[1:], pred)
    predictions = np.array(predictions).reshape(-1, 1)
    dummies = np.zeros((predictions.shape[0], lookback + 1))
    dummies[:, 0] = predictions.flatten()
    predictions = scaler.inverse_transform(dummies)[:, 0]
    return predictions


# Predict future dates
num_future_predictions = 2
last_known_data = X_test[-1].cpu().numpy()
future_predictions = predict_future_dates(
    model, last_known_data, scaler, num_future_predictions)

# Create future dates
last_date = test_results_df['Date'].iloc[-1]
future_dates = pd.date_range(
    start=last_date, periods=num_future_predictions + 1)[1:]

# Prepare the predictions dataframe
predictions = pd.DataFrame(
    future_predictions, index=future_dates, columns=['Predicted'])
predictions['Percent Change'] = predictions['Predicted'].pct_change()

print(predictions)
