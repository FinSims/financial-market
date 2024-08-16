import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class QuantPredictor:
    def __init__(self, lookback=5, hidden_size=16, num_stacked_layers=2, batch_size=16, learning_rate=0.005, num_epochs=30):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers, device):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers
            self.device = device
            self.gru = nn.GRU(input_size, hidden_size,
                              num_stacked_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_stacked_layers, batch_size,
                             self.hidden_size).to(self.device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    def prepare_dataframe_for_lstm(self, df):
        df = dc(df)
        for i in range(1, self.lookback + 1):
            df[f'Close(t-{i})'] = df['Close'].shift(i)

        # Add 3-day RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs = gain / loss
        df['RSI_3'] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)
        return df

    def generate_trading_signals(self, test_results_df, neutral_threshold=0.005):
        test_results_df['Signal'] = 'Neutral'
        test_results_df.loc[test_results_df['Predicted'] >
                            test_results_df['Actual'] * (1 + neutral_threshold), 'Signal'] = 'Buy'
        test_results_df.loc[test_results_df['Predicted'] <
                            test_results_df['Actual'] * (1 - neutral_threshold), 'Signal'] = 'Sell'
        return test_results_df

    def predict_stock(self, ticker):
        # Download historical stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")[['Close']]

        # Prepare data
        shifted_df = self.prepare_dataframe_for_lstm(df)
        shifted_df_np = shifted_df.to_numpy()
        data_scaler = MinMaxScaler(feature_range=(-1, 1))
        shifted_df_np = data_scaler.fit_transform(shifted_df_np)
        X = shifted_df_np[:, 1:]
        y = shifted_df_np[:, 0]
        X = dc(np.flip(X, axis=1))
        split_index = int(len(X) * .95)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        X_train = X_train.reshape((-1, self.lookback + 1, 1))  # +1 for RSI
        X_test = X_test.reshape((-1, self.lookback + 1, 1))  # +1 for RSI
        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()

        # Initialize model, loss function, and optimizer
        model = self.GRU(1, self.hidden_size,
                         self.num_stacked_layers, self.device).to(self.device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        grad_scaler = torch.cuda.amp.GradScaler()

        # Training the model
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            train_loader = DataLoader(self.TimeSeriesDataset(
                X_train, y_train), batch_size=self.batch_size, shuffle=True)
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(
                    self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(x_batch)
                    loss = loss_function(output, y_batch)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            test_loader = DataLoader(self.TimeSeriesDataset(
                X_test, y_test), batch_size=self.batch_size, shuffle=False)
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(
                        self.device), y_batch.to(self.device)
                    with torch.cuda.amp.autocast():
                        output = model(x_batch)
                        loss = loss_function(output, y_batch)
                    val_loss += loss.item()
            print(
                f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(test_loader)}')

        # Generate predictions on the test set
        with torch.no_grad():
            test_predictions = model(
                X_test.to(self.device)).detach().cpu().numpy().flatten()
        # +2 for Close and RSI
        dummies = np.zeros((X_test.shape[0], self.lookback + 2))
        dummies[:, 0] = test_predictions
        test_predictions = data_scaler.inverse_transform(dummies)[:, 0]
        dummies[:, 0] = y_test.flatten()
        new_y_test = data_scaler.inverse_transform(dummies)[:, 0]

        # Plot actual vs. predicted values for the test set
        plt.plot(new_y_test, label='Actual Close')
        plt.plot(test_predictions, label='Predicted Close')
        plt.xlabel('Day')
        plt.ylabel('Close')
        plt.legend()
        plt.show()

        # Generate test results DataFrame with dates, actual, and predicted values
        test_dates = df.index[-len(new_y_test):]
        test_results_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': new_y_test,
            'Predicted': test_predictions
        })

        # Generate trading signals for the test results
        test_results_df_with_signals = self.generate_trading_signals(
            test_results_df)
        print(test_results_df_with_signals)

        # Visualization of trading signals
        plt.figure(figsize=(14, 7))
        plt.plot(test_results_df['Date'], test_results_df['Actual'],
                 label='Actual Close', color='b')
        plt.plot(test_results_df['Date'], test_results_df['Predicted'],
                 label='Predicted Close', color='orange')
        buy_signals = test_results_df_with_signals[test_results_df_with_signals['Signal'] == 'Buy']
        sell_signals = test_results_df_with_signals[test_results_df_with_signals['Signal'] == 'Sell']
        neutral_signals = test_results_df_with_signals[test_results_df_with_signals['Signal'] == 'Neutral']

        plt.scatter(buy_signals['Date'], buy_signals['Actual'],
                    label='Buy Signal', marker='^', color='g')
        plt.scatter(sell_signals['Date'], sell_signals['Actual'],
                    label='Sell Signal', marker='v', color='r')
        plt.scatter(neutral_signals['Date'], neutral_signals['Actual'],
                    label='Neutral Signal', marker='o', color='gray')

        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(f'{ticker} Close Price with Trading Signals')
        plt.legend()
        plt.show()

        return test_results_df_with_signals


# Example usage:
# See how cucked it is with meta
# QCOM


# Just predict direction not how much, just predict direction for the next day.
predictor = QuantPredictor(lookback=5, hidden_size=64, num_stacked_layers=4,
                           batch_size=32, learning_rate=0.001, num_epochs=35)
result_df = predictor.predict_stock("NVDA")
print("Res")
print(result_df)
