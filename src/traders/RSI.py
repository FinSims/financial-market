import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates


class RelativeStrengthIndex:
    def __init__(self, stock, low=30, high=70, days=90, period=14):
        """
        Initializes a RelativeStrengthIndex object.
        """
        self.low = low
        self.high = high
        self.period = period
        self.stock = yf.download(
            stock, date.today() - timedelta(days + period), date.today())
        self.stock_name = stock

    def calculate_rsi_value(self):
        """
        Calculate the Relative Strength Index (RSI) value for a stock.
        """
        delta = self.stock['Close'].diff()

        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        avg_gain = gain[:self.period + 1].mean()
        avg_loss = loss[:self.period + 1].mean()

        gains_series = pd.Series(index=delta.index)
        losses_series = pd.Series(index=delta.index)

        gains_series.iloc[self.period] = avg_gain
        losses_series.iloc[self.period] = avg_loss

        for idx in range(self.period + 1, len(delta)):
            avg_gain = ((gains_series.iloc[idx - 1] * (self.period - 1) +
                         gain.iloc[idx]) / self.period)
            avg_loss = ((losses_series.iloc[idx - 1] * (self.period - 1) +
                         loss.iloc[idx]) / self.period)

            gains_series.iloc[idx] = avg_gain
            losses_series.iloc[idx] = avg_loss

        rs = gains_series / losses_series
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def display(self):
        """
        Displays the stock price and RSI indicator in a two-panel chart.
        """
        plt.style.use('seaborn-v0_8')

        # Calculate RSI and find first valid RSI value
        rsi_values = self.calculate_rsi_value()
        first_valid_index = rsi_values.first_valid_index()

        # Trim both price and RSI data to start from first valid RSI value
        self.stock = self.stock[first_valid_index:]
        self.stock['RSI'] = rsi_values[first_valid_index:]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])

        # Add spacing between subplots
        plt.subplots_adjust(hspace=0.5)

        # Price subplot
        ax1.plot(self.stock.index, self.stock['Close'], linewidth=1.5,
                 color='blue', label='Price')
        ax1.set_title(f'{self.stock_name} Price', fontsize=12, pad=15)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        # RSI subplot
        ax2.plot(self.stock.index, self.stock['RSI'], color='purple',
                 linewidth=1.5, label='RSI')
        ax2.axhline(self.high, linestyle='--', linewidth=1, color='red',
                    alpha=0.5, label='Overbought')
        ax2.axhline(self.low, linestyle='--', linewidth=1, color='green',
                    alpha=0.5, label='Oversold')

        # Add colored regions
        ax2.fill_between(self.stock.index, self.low, self.stock['RSI'],
                         where=(self.stock['RSI'] <= self.low),
                         color='green', alpha=0.3)
        ax2.fill_between(self.stock.index, self.high, self.stock['RSI'],
                         where=(self.stock['RSI'] >= self.high),
                         color='red', alpha=0.3)

        # Set y-limits for RSI
        ax2.set_ylim(0, 100)

        # Format dates for both subplots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add RSI title with padding
        ax2.set_title(f'{self.stock_name} RSI ({self.period})', fontsize=12, pad=15)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # Ensure layout is tight after all elements are added
        plt.tight_layout()

        return fig

    def get_current_rsi(self):
        """
        Returns the most recent RSI value.
        """
        rsi = self.calculate_rsi_value()
        return rsi.iloc[-1]


import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates


class SimpleMovingAverage:
    def __init__(self, stock, period=10, days=90):
        # Add period to days to account for SMA calculation
        self.stock = yf.download(
            stock, date.today() - timedelta(days + period), date.today())
        self.period = period
        self.stock_name = stock

    def calculate_sma(self):
        """
        Calculate Simple Moving Average for the stock.
        """
        return self.stock['Close'].rolling(window=self.period).mean()

    def display(self):
        """
        Displays the stock price with MA overlay.
        """
        plt.style.use('seaborn-v0_8')

        # Calculate SMA and find first valid value
        self.stock["MA"] = self.calculate_sma()
        first_valid_index = self.stock["MA"].first_valid_index()

        # Trim both price and MA data to start from first valid MA value
        trimmed_data = self.stock[first_valid_index:]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price and MA
        ax.plot(trimmed_data.index, trimmed_data['Close'], linewidth=1.5,
                color='blue', label='Price')
        ax.plot(trimmed_data.index, trimmed_data['MA'], linewidth=1.5,
                color='red', label=f'MA ({self.period})')

        # Formatting
        ax.set_title(f'{self.stock_name} Price with {self.period}-day Moving Average',
                     fontsize=12, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Format dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Ensure layout is tight
        plt.tight_layout()

        return fig

    def get_current_sma(self):
        """
        Returns the most recent SMA value.
        """
        sma = self.calculate_sma()
        return sma.iloc[-1]

    def get_signals(self):
        """
        Returns buy/sell signals based on price crossing the MA.
        """
        self.stock['MA'] = self.calculate_sma()
        first_valid_index = self.stock["MA"].first_valid_index()
        trimmed_data = self.stock[first_valid_index:]

        # Create signals when price crosses MA
        signals = pd.DataFrame(index=trimmed_data.index)
        signals['Signal'] = 0

        # Price crosses above MA (buy signal)
        signals.loc[(trimmed_data['Close'] > trimmed_data['MA']) &
                    (trimmed_data['Close'].shift(1) <= trimmed_data['MA'].shift(1)), 'Signal'] = 1

        # Price crosses below MA (sell signal)
        signals.loc[(trimmed_data['Close'] < trimmed_data['MA']) &
                    (trimmed_data['Close'].shift(1) >= trimmed_data['MA'].shift(1)), 'Signal'] = -1

        return signals

    def display_with_signals(self):
        """
        Displays the stock price with MA overlay and buy/sell signals.
        """
        plt.style.use('seaborn-v0_8')

        # Calculate SMA and find first valid value
        self.stock["MA"] = self.calculate_sma()
        first_valid_index = self.stock["MA"].first_valid_index()

        # Trim both price and MA data to start from first valid MA value
        trimmed_data = self.stock[first_valid_index:]
        signals = self.get_signals()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price and MA
        ax.plot(trimmed_data.index, trimmed_data['Close'], linewidth=1.5,
                color='blue', label='Price')
        ax.plot(trimmed_data.index, trimmed_data['MA'], linewidth=1.5,
                color='red', label=f'MA ({self.period})')

        # Plot buy signals
        buy_signals = signals[signals['Signal'] == 1]
        ax.scatter(buy_signals.index, trimmed_data.loc[buy_signals.index, 'Close'],
                   marker='^', color='green', s=100, label='Buy Signal')

        # Plot sell signals
        sell_signals = signals[signals['Signal'] == -1]
        ax.scatter(sell_signals.index, trimmed_data.loc[sell_signals.index, 'Close'],
                   marker='v', color='red', s=100, label='Sell Signal')

        # Formatting
        ax.set_title(f'{self.stock_name} Price with {self.period}-day Moving Average',
                     fontsize=12, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Format dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Ensure layout is tight
        plt.tight_layout()

        return fig


class ExponentialMovingAverage:
    def __init__(self, stock, period=10, days=90):
        self.stock = yf.download(
            stock, date.today() - timedelta(days), date.today())
        self.period = period
        self.stock_name = stock

    def calculate_ema_helper(self, end, index):
        k = 2/(self.period+1)
        ema = self.stock['Close'][index]
        if index == 0:
            return ema
        else:
            ema = ema * k + self.calculate_ema_helper(end, index - 1) * (1 - k)
            print(ema)
        return ema

    def calculate_ema(self):
        stock = self.stock['Close']
        stocks = []
        self.stock['Close'].dropna(inplace=True)
        for end in range(len(stock)):
            print(end)

            stocks.append(self.calculate_ema_helper(end, end))
        print(len(stocks))
        self.stock["EMA"] = stocks
        return self.stock["EMA"]

    def display(self):
        self.calculate_ema()

        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
        ax1.plot(self.stock['Close'], linewidth=2.5)

        ax1.set_title(f'{self.stock_name}')
        ax2.plot(self.stock['EMA'], color='red', linewidth=1.5)
        ax2.set_title(f'{self.stock_name} EMA')


if __name__ == "__main__":
    '''rsi = RelativeStrengthIndex("PLTR")

    rsi.display_rsi()
    plt.show()

    sma = SimpleMovingAverage("PLTR")
    sma.display()
    plt.show()'''

    sma = SimpleMovingAverage("NKE")
    sma.display()
    plt.show()
