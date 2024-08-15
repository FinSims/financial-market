import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class TechnicalTrader:
    def __init__(self, ticker: str, period: str = "5y", ma_window: int = 20, bb_window: int = 20, bb_std: float = 2.0):
        self.ticker = ticker
        self.period = period
        self.ma_window = ma_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.df = self._get_data()
        self._calculate_indicators()

    def _get_data(self) -> pd.DataFrame:
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=self.period)
        return df

    def _calculate_indicators(self):
        # Calculate Moving Average
        self.df['MA'] = self.df['Close'].rolling(window=self.ma_window).mean()

        # Calculate Bollinger Bands
        self.df['BB_Middle'] = self.df['Close'].rolling(
            window=self.bb_window).mean()
        self.df['BB_Std'] = self.df['Close'].rolling(
            window=self.bb_window).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + \
            (self.bb_std * self.df['BB_Std'])
        self.df['BB_Lower'] = self.df['BB_Middle'] - \
            (self.bb_std * self.df['BB_Std'])

        # Calculate percent difference from MA for mean reversion
        self.df['Percent_From_MA'] = (
            self.df['Close'] - self.df['MA']) / self.df['MA'] * 100

        # Drop NaN values
        self.df.dropna(inplace=True)

    def generate_signals(self, bb_threshold: float = 1.0, mr_threshold: float = 5.0) -> pd.DataFrame:
        self.df['Signal'] = 'Hold'

        # Bollinger Bands signals
        self.df.loc[self.df['Close'] > self.df['BB_Upper']
                    * (1 + bb_threshold/100), 'Signal'] = 'Sell'
        self.df.loc[self.df['Close'] < self.df['BB_Lower']
                    * (1 - bb_threshold/100), 'Signal'] = 'Buy'

        # Mean Reversion signals
        self.df.loc[self.df['Percent_From_MA']
                    > mr_threshold, 'Signal'] = 'Sell'
        self.df.loc[self.df['Percent_From_MA']
                    < -mr_threshold, 'Signal'] = 'Buy'

        return self.df

    def generate_ml_signals(self):
        # Prepare features
        features = ['MA', 'BB_Upper', 'BB_Lower', 'Percent_From_MA']
        X = self.df[features]

        # Prepare target: 1 if price increased, 0 if decreased
        y = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)

        # Remove last row as we don't have a target for it
        X = X.iloc[:-1]
        y = y.iloc[:-1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        # Add predictions to the dataframe
        self.df['ML_Signal'] = pd.Series(predictions, index=X.index)
        self.df['ML_Signal'] = self.df['ML_Signal'].map({1: 'Buy', 0: 'Sell'})

        return self.df

    def plot_indicators(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, figsize=(15, 20), sharex=True)

        # Plot 1: Bollinger Bands
        ax1.plot(self.df.index, self.df['Close'], label='Close Price')
        ax1.plot(self.df.index, self.df['BB_Upper'],
                 label='Upper BB', linestyle='--')
        ax1.plot(self.df.index, self.df['BB_Middle'],
                 label='Middle BB', linestyle='--')
        ax1.plot(self.df.index, self.df['BB_Lower'],
                 label='Lower BB', linestyle='--')
        ax1.set_title('Bollinger Bands')
        ax1.legend()

        # Plot 2: Moving Average
        ax2.plot(self.df.index, self.df['Close'], label='Close Price')
        ax2.plot(self.df.index, self.df['MA'],
                 label=f'{self.ma_window}-day MA')
        ax2.set_title('Moving Average')
        ax2.legend()

        # Plot 3: Bollinger Bands and Moving Average
        ax3.plot(self.df.index, self.df['Close'], label='Close Price')
        ax3.plot(self.df.index, self.df['BB_Upper'],
                 label='Upper BB', linestyle='--')
        ax3.plot(self.df.index, self.df['BB_Lower'],
                 label='Lower BB', linestyle='--')
        ax3.plot(self.df.index, self.df['MA'],
                 label=f'{self.ma_window}-day MA')
        ax3.set_title('Bollinger Bands and Moving Average')
        ax3.legend()

        # Plot 4: Trading Signals
        ax4.plot(self.df.index, self.df['Close'], label='Close Price')
        ax4.scatter(self.df[self.df['Signal'] == 'Buy'].index,
                    self.df[self.df['Signal'] == 'Buy']['Close'],
                    label='Buy Signal', marker='^', color='g')
        ax4.scatter(self.df[self.df['Signal'] == 'Sell'].index,
                    self.df[self.df['Signal'] == 'Sell']['Close'],
                    label='Sell Signal', marker='v', color='r')
        ax4.set_title('Trading Signals')
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def plot_ml_signals(self):
        if 'ML_Signal' not in self.df.columns:
            self.generate_ml_signals()

        plt.figure(figsize=(15, 7))
        plt.plot(self.df.index, self.df['Close'], label='Close Price')

        buy_signals = self.df[self.df['ML_Signal'] == 'Buy']
        sell_signals = self.df[self.df['ML_Signal'] == 'Sell']

        plt.scatter(buy_signals.index, buy_signals['Close'],
                    label='Buy Signal', marker='^', color='g', alpha=0.7)
        plt.scatter(sell_signals.index, sell_signals['Close'],
                    label='Sell Signal', marker='v', color='r', alpha=0.7)

        plt.title('ML-Generated Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_all_signals(self):
        if 'ML_Signal' not in self.df.columns:
            self.generate_ml_signals()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), sharex=True)

        # Plot 1: Technical Indicators Signals
        ax1.plot(self.df.index, self.df['Close'], label='Close Price')
        ax1.scatter(self.df[self.df['Signal'] == 'Buy'].index,
                    self.df[self.df['Signal'] == 'Buy']['Close'],
                    label='Technical Buy Signal', marker='^', color='g', alpha=0.7)
        ax1.scatter(self.df[self.df['Signal'] == 'Sell'].index,
                    self.df[self.df['Signal'] == 'Sell']['Close'],
                    label='Technical Sell Signal', marker='v', color='r', alpha=0.7)
        ax1.set_title('Technical Indicators Signals')
        ax1.legend()

        # Plot 2: ML-Generated Signals
        ax2.plot(self.df.index, self.df['Close'], label='Close Price')
        ax2.scatter(self.df[self.df['ML_Signal'] == 'Buy'].index,
                    self.df[self.df['ML_Signal'] == 'Buy']['Close'],
                    label='ML Buy Signal', marker='^', color='g', alpha=0.7)
        ax2.scatter(self.df[self.df['ML_Signal'] == 'Sell'].index,
                    self.df[self.df['ML_Signal'] == 'Sell']['Close'],
                    label='ML Sell Signal', marker='v', color='r', alpha=0.7)
        ax2.set_title('ML-Generated Signals')
        ax2.legend()

        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()

    def plot_bollinger_bands_with_signals(self):
        plt.figure(figsize=(15, 10))

        # Plot Close price and Bollinger Bands
        plt.plot(self.df.index, self.df['Close'],
                 label='Close Price', color='blue')
        plt.plot(self.df.index, self.df['BB_Upper'],
                 label='Upper BB', color='gray', linestyle='--')
        plt.plot(self.df.index, self.df['BB_Middle'],
                 label='Middle BB', color='gray', linestyle='-')
        plt.plot(self.df.index, self.df['BB_Lower'],
                 label='Lower BB', color='gray', linestyle='--')

        # Plot Buy and Sell signals
        plt.scatter(self.df[self.df['Signal'] == 'Buy'].index,
                    self.df[self.df['Signal'] == 'Buy']['Close'],
                    label='Buy Signal', marker='^', color='g', s=100)
        plt.scatter(self.df[self.df['Signal'] == 'Sell'].index,
                    self.df[self.df['Signal'] == 'Sell']['Close'],
                    label='Sell Signal', marker='v', color='r', s=100)

        plt.title(f'{self.ticker} - Bollinger Bands with Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def backtest(self) -> Tuple[float, float]:
        self.df['Position'] = 0
        self.df.loc[self.df['Signal'] == 'Buy', 'Position'] = 1
        self.df.loc[self.df['Signal'] == 'Sell', 'Position'] = -1

        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Strategy_Returns'] = self.df['Position'].shift(
            1) * self.df['Returns']

        cumulative_returns = (1 + self.df['Returns']).cumprod()
        cumulative_strategy_returns = (
            1 + self.df['Strategy_Returns']).cumprod()

        total_return = cumulative_returns.iloc[-1] - 1
        strategy_return = cumulative_strategy_returns.iloc[-1] - 1

        return total_return, strategy_return

    def plot_returns(self):
        cumulative_returns = (1 + self.df['Returns']).cumprod()
        cumulative_strategy_returns = (
            1 + self.df['Strategy_Returns']).cumprod()

        plt.figure(figsize=(15, 7))
        plt.plot(self.df.index, cumulative_returns,
                 label='Buy and Hold Returns')
        plt.plot(self.df.index, cumulative_strategy_returns,
                 label='Strategy Returns')
        plt.title('Cumulative Returns: Buy and Hold vs Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.show()

    def create_final_df(self):
        if 'ML_Signal' not in self.df.columns:
            self.generate_ml_signals()

        final_df = self.df[['Close', 'Signal', 'ML_Signal']].copy()
        final_df['Returns'] = self.df['Close'].pct_change()
        final_df['Strategy_Returns'] = final_df['Returns'] * \
            final_df['Signal'].map({'Buy': 1, 'Sell': -1, 'Hold': 0}).shift(1)

        return final_df


# Example usage
trader = TechnicalTrader("TSLA")
trader.generate_signals()
trader.plot_bollinger_bands_with_signals()

total_return, strategy_return = trader.backtest()
print(f"Buy and Hold Return: {total_return:.2%}")
print(f"Strategy Return: {strategy_return:.2%}")

final_df = trader.create_final_df()
print(final_df.head(10))
print(final_df.tail(10))
