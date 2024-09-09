import utilities.Trader as Trader

import yfinance as yf
import matplotlib.pyplot as plt
from datetime import *


class RelativeStrengthIndex:
    pass

    def __init__(self, stock, low=30, high=70, days=90):
        """
        Initializes a RelativeStrengthIndex object.

        """
        self.low = low
        self.high = high
        self.stock = yf.download(
            stock, date.today() - timedelta(days), date.today())
        self.stock_name = stock

    def find_average_gains(self):
        """
        Calculates the average gains of a stock's Close price over a given period.
        """
        delta = self.stock['Close'].diff(1).dropna()
        gains = delta.copy()
        gains[gains < 0] = 0

        average_gains = abs(gains.ewm(com=14 - 1, adjust=False).mean())

        return average_gains

    def find_average_losses(self):
        """
        Calculates the average losses of a stock's Close price over a given period.
        """
        delta = self.stock['Close'].diff(1).dropna()
        losses = delta.copy()
        losses[losses > 0] = 0

        average_losses = abs(losses.ewm(com=14 - 1, adjust=False).mean())

        return average_losses

    def calculate_rsi_value(self):
        """
        Calculate the Relative Strength Index (RSI) value for a stock.

        This method calculates the RSI value for a stock using the average gains and average losses of its Close price over a given period. 
        The RSI value is calculated using the formula:

        RSI = 100 - (100 / (1 + (average_gains / average_losses)))

        The average gains and average losses are calculated using the `find_average_gains` and `find_average_losses` methods respectively.

        The method first prints the average gains and average losses using the `find_average_gains` and `find_average_losses` methods. 
        Then, it calculates the RSI value using the formula and returns it.
        """
        # 100 - [100/ 1+ ((average gains / average losses)]
        print(self.find_average_gains())
        print(self.find_average_losses())
        calculate_rsi = 100 - \
            (100 / (1 + (self.find_average_gains() / self.find_average_losses())))

        return calculate_rsi

    def display_rsi(self):
        """
        Displays the Relative Strength Index (RSI) of a stock.

        This method calculates the RSI value for the stock and displays it
        using matplotlib. The RSI is calculated using the formula:
        100 - (100 / (1 + (average gains / average losses))).
        """

        self.stock["RSI"] = self.calculate_rsi_value()
        # Increase the width to 12 inches and height to 6 inches
        plt.figure(figsize=(12, 6))

        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
        ax1.plot(self.stock['Close'], linewidth=2.5)
        ax1.set_title(f'{self.stock_name}')
        ax2.plot(self.stock['RSI'], color='red', linewidth=1.5)
        ax2.axhline(30, linestyle='--', linewidth=1.5, color='grey')
        ax2.axhline(70, linestyle='--', linewidth=1.5, color='grey')
        ax2.set_title(f'{self.stock_name} RSI')


class SimpleMovingAverage:
    def __init__(self, stock, period=10, days=90):
        self.stock = yf.download(
            stock, date.today() - timedelta(days), date.today())
        self.period = period
        self.stock_name = stock

    def calculate_sma(self):
        delta = self.stock['Close']

        average_losses = delta.ewm(com=self.period - 1, adjust=False).mean()

        return average_losses

    def display(self):
        self.stock["MA"] = self.calculate_sma()

        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
        ax1.plot(self.stock['Close'], linewidth=2.5)

        ax1.set_title(f'{self.stock_name}')
        ax2.plot(self.stock['MA'], color='red', linewidth=1.5)
        ax2.set_title(f'{self.stock_name} MA')


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

    ema = ExponentialMovingAverage("NKE")
    ema.display()
    plt.show()
