import Trader

import yfinance as yf
import matplotlib.pyplot as plt
from datetime import *
import talib


class RelativeStrengthIndex:
    pass
    def __init__(self, stock, low = 30, high = 70):
        self.low = low
        self.high = high
        self.stock = yf.download(stock,date.today() - timedelta(days=90),date.today())
        self.stock_name = stock

    def find_average_gains(self):
        delta = self.stock['Close'].diff(1).dropna()
        gains = delta.copy()
        gains[gains < 0] = 0
        
        average_gains = abs(gains.ewm(com=14 - 1, adjust=False).mean())
        
        return average_gains
        
    
    def find_average_losses(self):
        delta = self.stock['Close'].diff(1).dropna()
        losses = delta.copy()
        losses[losses > 0] = 0
        
        average_losses = abs(losses.ewm(com=14 - 1, adjust=False).mean())
        
        return average_losses
    
    def calculate_rsi_value(self):
        # 100 - [100/ 1+ ((average gains / average losses)]
        print(self.find_average_gains())
        print(self.find_average_losses())
        calculate_rsi = 100 - (100 / (1 + (self.find_average_gains() / self.find_average_losses())))
        
        return calculate_rsi
    
    def display_rsi(self):
        reversed_df = self.stock.iloc[::-1]
        self.stock["RSI"] = self.calculate_rsi_value()
        
        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
        ax1.plot(self.stock['Close'], linewidth=2.5)
        ax1.set_title(f'{self.stock_name}')
        ax2.plot(self.stock['RSI'], color='red', linewidth=1.5)
        ax2.axhline(30, linestyle='--', linewidth=1.5, color='grey')
        ax2.axhline(70, linestyle='--', linewidth=1.5, color='grey')
        ax2.set_title(f'{self.stock_name} RSI')
        
if __name__ == "__main__":
    rsi = RelativeStrengthIndex("PLTR")
    
    rsi.display_rsi()
    plt.show()