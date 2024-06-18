import pandas as pd
from tabulate import tabulate
from datetime import datetime
import time
import uuid


class Security:
    _instances = {}

    def __new__(cls, symbol, *args, **kwargs):
        if symbol not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[symbol] = instance
        return cls._instances[symbol]

    def __init__(self, ticker):
        if not hasattr(self, 'initialized'):
            self.ticker = ticker
            self.buy_orders = []
            self.sell_orders = []
            self.trade_history = []
            self.bid = 0
            self.ask = 0
            self.last = 0

    @classmethod
    def get_instance(cls, symbol):
        return cls._instances.get(symbol, None)

    def __update_bid_ask(self):
        self.buy_orders.sort(key=lambda order: order["price"], reverse=True)  # sort from highest to lowest
        self.sell_orders.sort(key=lambda order: order["price"], reverse=False)  # sort from lowest to highest

        if len(self.buy_orders) > 0:
            self.bid = max(self.buy_orders, key=lambda x: x['price'])
        else:
            self.bid = 0

        if len(self.sell_orders) > 0:
            self.ask = min(self.sell_orders, key=lambda x: x['price'])
        else:
            self.ask = 0

    def execute_market_order(self, trade):
        # buy_orders and sell_orders are sorted appropriately
        self.__update_bid_ask()

        if trade["side"] == "buy":  # market buy
            sell_index = 0

            while sell_index < len(self.sell_orders):
                sell_order = self.sell_orders[sell_index]

                if sell_order is None:
                    break

                trade_price = sell_order["price"]
                trade_quantity = min(trade["size"], sell_order["size"])
                self.trade_history.append(
                    {"time": datetime.now(), "type": "buy", "price": trade_price, "quantity": trade_quantity})
                trade["size"] -= trade_quantity
                sell_order["size"] -= trade_quantity

                # If the buy market order isn't fully filled yet, we'll move up the sellers
                if sell_order["size"] == 0 and trade["size"] > 0:
                    sell_index += 1
                else:
                    break
        else:  # market sell
            buy_index = 0

            while buy_index < len(self.buy_orders):
                buy_order = self.buy_orders[buy_index]

                if buy_order is None:
                    break

                trade_price = buy_order["price"]
                trade_quantity = min(buy_order["size"], trade["size"])
                self.trade_history.append(
                    {"time": datetime.now(), "type": "sell", "price": trade_price, "quantity": trade_quantity})
                buy_order["size"] -= trade_quantity
                trade["size"] -= trade_quantity

                # If the sell market order isn't fully filled yet, we'll move down the buyers
                if buy_order["size"] == 0 and trade["size"] > 0:
                    buy_index += 1
                else:
                    break

    def execute_limit_order(self):
        # buy_orders and sell_orders are sorted appropriately
        self.__update_bid_ask()

        buy_index = 0
        sell_index = 0

        if self.buy_orders is None or self.sell_orders is None:
            return

        while buy_index < len(self.buy_orders) and sell_index < len(self.sell_orders):
            buy_order = self.buy_orders[buy_index]
            sell_order = self.sell_orders[sell_index]

            if buy_order is None or sell_order is None:
                break

            # If the buy order is greater than or equal than the sell order
            if buy_order["price"] >= sell_order["price"]:
                # If buy order was submitted earlier than sell order, we'll use buy order's price
                if buy_order["time"] < sell_order["time"]:
                    trade_price = buy_order["price"]
                else:
                    # If sell order was submitted earlier than buy order, we'll use sell order's price
                    trade_price = sell_order["price"]

                trade_quantity = min(buy_order["size"], sell_order["size"])
                self.trade_history.append(
                    {"time": datetime.now(), "type": "buy", "price": trade_price, "quantity": trade_quantity})
                buy_order["size"] -= trade_quantity
                sell_order["size"] -= trade_quantity

                # If this buy order is overpaying, and they still have shares left, there is a possibility they can
                # get the rest of their shares filled, so we'll move up the sellers list
                if sell_order["size"] == 0 and buy_order["size"] > 0 and buy_order["price"] > sell_order["price"]:
                    # Note that this may not be filled if it does not meet the limit criteria of the seller
                    sell_index += 1
                else:
                    break
            else:
                break

    def display_order_book(self):
        return self.buy_orders + self.sell_orders

    def display_trade_history(self, ascending):
        if ascending:
            self.trade_history.sort(key=lambda x: x["time"])
        else:
            self.trade_history.sort(key=lambda x: x["time"], reverse=True)

        return self.trade_history


class Trader:
    def __init__(self):
        self.id = uuid.uuid4()

    def create_market_order(self, ticker, side, quantity):
        instrument = Security.get_instance(ticker)

        if instrument is None:
            instrument = Security(ticker)
        instrument.__update_bid_ask()

        if side == "buy":
            trade = {
                "trader": self.id,
                "time": datetime.now(),
                "ticker": ticker,
                "side": side,
                "price": instrument.sell_orders[0]["price"],
                "size": quantity,
                "type": "market"
            }

            instrument.execute_market_order(trade)
        else:
            trade = {
                "trader": self.id,
                "time": datetime.now(),
                "ticker": ticker,
                "side": side,
                "price": instrument.buy_orders[0]["price"],
                "size": quantity,
                "type": "market"
            }

            instrument.execute_market_order(trade)

    def create_limit_order(self, ticker, side, price, quantity):
        instrument = Security.get_instance(ticker)

        if instrument is None:
            instrument = Security(ticker)

        # Finds first order in which side is same and price is same, so we can add on the quantity if it exists
        if side == "buy":
            matching_order_index = next(
                (i for i, order in enumerate(instrument.buy_orders) if order["price"] == price), None)

            if matching_order_index:
                instrument.buy_orders[matching_order_index]["size"] += quantity
            else:
                instrument.buy_orders.append({
                    "trader": self.id,
                    "time": datetime.now(),
                    "ticker": ticker,
                    "side": side,
                    "price": price,
                    "size": quantity,
                    "type": "limit"
                })
        else:
            matching_order_index = next(
                (i for i, order in enumerate(instrument.sell_orders) if order["price"] == price), None)

            if matching_order_index:
                instrument.sell_orders[matching_order_index]["size"] += quantity
            else:
                instrument.sell_orders.append({
                    "trader": self.id,
                    "time": datetime.now(),
                    "ticker": ticker,
                    "side": side,
                    "price": price,
                    "size": quantity,
                    "type": "limit"
                })

        instrument.execute_limit_order()


my_stock = Security("AAPL")
trader = Trader()
# print(my_stock.order_book)
trader.create_limit_order("AAPL", "buy", 102.0, 17)
time.sleep(1)
trader.create_limit_order("AAPL", "sell", 101.0, 7)
time.sleep(1)
trader.create_limit_order("AAPL", "sell", 101.0, 2)
time.sleep(1)
trader.create_limit_order("AAPL", "sell", 100.0, 2)
time.sleep(1)
order_book = my_stock.display_order_book()
trade_history = my_stock.display_trade_history(False)

ob_df = pd.DataFrame(order_book)
ob_table = tabulate(ob_df, headers='keys', tablefmt='fancy_grid')

th_df = pd.DataFrame(trade_history)
th_table = tabulate(th_df, headers='keys', tablefmt='fancy_grid')

print("ORDER BOOK:")
print(ob_table)

print("TRADE HISTORY:")
print(th_table)
