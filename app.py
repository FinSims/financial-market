import pandas as pd
from tabulate import tabulate
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from Security import Security
from Trader import Trader
from InstitutionalTrader import InstitutionalTrader


def plot_security_prices(security):
    timestamps = [entry["timestamp"] for entry in security.last]
    prices = [entry["price"] for entry in security.last]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, prices, marker='o')
    plt.title(f"Price History of {security.ticker}")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ! Important: Bug with calculating average price when shorting and covering back shares
my_stock = Security("AAPL")
trader = Trader(False, 11000)
another_trader = Trader(True, 13000)
inst_trader = InstitutionalTrader(10000000000)
# print(my_stock.order_book)
trader.create_limit_order("AAPL", "buy", 102.0, 17)
time.sleep(1)
another_trader.create_limit_order("AAPL", "sell", 101.0, 7)
time.sleep(1)
trader.create_limit_order("AAPL", "buy", 103.0, 7)
time.sleep(1)
another_trader.create_limit_order("AAPL", "sell", 103.0, 7)
time.sleep(1)
trader.create_limit_order("AAPL", "buy", 105.0, 7)
time.sleep(1)
another_trader.create_limit_order("AAPL", "sell", 104.0, 7000)
time.sleep(1)
inst_trader.generate_trade_signal("AAPL", 90)
# time.sleep(1)
# trader.create_limit_order("AAPL", "sell", 101.0, 2)
# time.sleep(1)
# trader.create_limit_order("AAPL", "sell", 100.0, 2)
# time.sleep(1)
order_book = my_stock.display_order_book()
trade_history = another_trader.display_trade_history(False)
portfolio = another_trader.portfolio
# print(my_stock.ask)

print("Buyer's balance:", trader.balance)
print("Seller's balance:", another_trader.balance)

ob_df = pd.DataFrame(order_book)
ob_table = tabulate(ob_df, headers='keys', tablefmt='fancy_grid')

th_df = pd.DataFrame(trade_history)
th_table = tabulate(th_df, headers='keys', tablefmt='fancy_grid')

port_df = pd.DataFrame(portfolio)
port_table = tabulate(port_df, headers='keys', tablefmt='fancy_grid')

print("\nORDER BOOK:")
print(ob_table)

print("TRADE HISTORY:")
print(th_table)

print("PORTFOLIO:")
print(port_table)

print(my_stock.last)

plot_security_prices(my_stock)
