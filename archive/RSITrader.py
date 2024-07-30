
import yfinance as yf
import talib
import matplotlib.pyplot as plt
import datetime as dt
# Get symbol OHLC data
data = yf.download("PLTR", start=dt.date.today() - dt.timedelta(days = 90), end=dt.date.today(), interval = "1d")


def RSI(data, window=14, adjust=False):
    delta = data['Close'].diff(1).dropna()
    loss = delta.copy()
    gains = delta.copy()

    gains[gains < 0] = 0
    loss[loss > 0] = 0

    gain_ewm = gains.ewm(com=window - 1, adjust=adjust).mean()
    loss_ewm = abs(loss.ewm(com=window - 1, adjust=adjust).mean())

    RS = gain_ewm / loss_ewm
    RSI = 100 - 100 / (1 + RS)

    return RSI
    
reversed_df = data.iloc[::-1]
data["RSI"] = talib.RSI(reversed_df["Close"], 14)
print(data.head())

ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=4, colspan=1)
ax1.plot(data['Close'], linewidth=2.5)
ax1.set_title('Bitcoin USD (BTC-USD)')
ax2.plot(data['RSI'], color='red', linewidth=1.5)
ax2.axhline(30, linestyle='--', linewidth=1.5, color='grey')
ax2.axhline(70, linestyle='--', linewidth=1.5, color='grey')
ax2.set_title('Bitcoin RSI')

plt.show()