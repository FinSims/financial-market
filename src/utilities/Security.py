from datetime import datetime, time
import yfinance as yf
from .Trader import Trader


class Security:
    _instances = {}

    # Ensures when creating a new instance that the ticker is not already created
    def __new__(cls, symbol, *args, **kwargs):
        if symbol not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[symbol] = instance
        return cls._instances[symbol]

    def __init__(self, ticker):
        if not hasattr(self, 'initialized'):
            stock = yf.Ticker(ticker)

            self.ticker = ticker
            self.buy_orders = []
            self.sell_orders = []
            self.bid = stock.info["bid"]
            self.ask = stock.info["ask"]

            now = datetime.now()
            market_open = time(9, 30)
            market_close = time(16, 0)

            if market_open <= now.time() <= market_close:
                self.last = [{
                    "timestamp": now,
                    "price": stock.info["regularMarketPreviousClose"]
                }]
            else:
                self.last = [{
                    "timestamp": now.replace(hour=16, minute=0, second=0, microsecond=0),
                    "price": stock.info["regularMarketPreviousClose"]
                }]

    @classmethod
    def get_instance(cls, symbol):
        return cls._instances.get(symbol, None)

    def update_bid_ask(self):
        # sort from highest to lowest
        self.buy_orders.sort(key=lambda order: order["price"], reverse=True)
        # sort from lowest to highest
        self.sell_orders.sort(key=lambda order: order["price"], reverse=False)

        if len(self.buy_orders) > 0:
            self.bid = max(self.buy_orders, key=lambda x: x['price'])["price"]
        else:
            self.bid = 0

        if len(self.sell_orders) > 0:
            self.ask = min(self.sell_orders, key=lambda x: x['price'])["price"]
        else:
            self.ask = 0

    def execute_market_order(self, trade):
        # buy_orders and sell_orders are sorted appropriately
        self.update_bid_ask()

        if trade["side"] == "buy":  # market buy
            sell_index = 0

            filtered_sell_orders = [
                item for item in self.sell_orders if item.get("trader") != trade["trader"]]
            while sell_index < len(self.sell_orders):
                # sell_order = self.sell_orders[sell_index]
                sell_order = filtered_sell_orders[sell_index]

                if sell_order is None:
                    break

                trade_price = sell_order["price"]
                trade_quantity = min(trade["size"], sell_order["size"])

                buy_trader = Trader.search_by_id(trade["trader"])
                sell_trader = Trader.search_by_id(sell_order["trader"])

                for i in range(trade_quantity):
                    trade["size"] -= 1
                    sell_order["size"] -= 1

                    result = buy_trader.transaction(
                        "buy", trade["ticker"], trade_price, trade_quantity)

                    if result is False:
                        trade_quantity = i
                        break

                    sell_trader.transaction(
                        "sell", trade["ticker"], trade_price, trade_quantity)

                timestamp = datetime.now()

                buy_trader.update_trade_history(
                    trade["ticker"], timestamp, "buy", trade_quantity, trade_price)
                sell_trader.update_trade_history(
                    sell_order["ticker"], timestamp, "buy", trade_quantity, trade_price)

                self.last.append({
                    "timestamp": timestamp,
                    "price": trade_price
                })

                # If the buy market order isn't fully filled yet, we'll move up the sellers
                if sell_order["size"] == 0 and trade["size"] > 0:
                    sell_index += 1
                else:
                    break
        else:  # market sell
            buy_index = 0

            filtered_buy_orders = [
                item for item in self.buy_orders if item.get("trader") != trade["trader"]]
            while buy_index < len(self.buy_orders):
                # buy_order = self.buy_orders[buy_index]
                buy_order = filtered_buy_orders[buy_index]

                if buy_order is None:
                    break

                trade_price = buy_order["price"]
                trade_quantity = min(buy_order["size"], trade["size"])

                buy_trader = Trader.search_by_id(buy_order["trader"])
                sell_trader = Trader.search_by_id(trade["trader"])

                for i in range(trade_quantity):
                    trade["size"] -= 1
                    buy_order["size"] -= 1

                    result = buy_trader.transaction(
                        "buy", trade["ticker"], trade_price, trade_quantity)

                    if result is False:
                        trade_quantity = i
                        break

                    sell_trader.transaction(
                        "sell", trade["ticker"], trade_price, trade_quantity)

                timestamp = datetime.now()
                buy_trader.update_trade_history(
                    buy_order["ticker"], timestamp, "buy", trade_quantity, trade_price)
                sell_trader.update_trade_history(
                    trade["ticker"], timestamp, "sell", trade_quantity, trade_price)

                self.last.append({
                    "timestamp": timestamp,
                    "price": trade_price
                })

                # If the sell market order isn't fully filled yet, we'll move down the buyers
                if buy_order["size"] == 0 and trade["size"] > 0:
                    buy_index += 1
                else:
                    break

    def execute_limit_order(self, side):
        # buy_orders and sell_orders are sorted appropriately
        self.update_bid_ask()

        buy_index = 0
        sell_index = 0

        if self.buy_orders is None or self.sell_orders is None:
            return

        while buy_index < len(self.buy_orders) and sell_index < len(self.sell_orders):
            buy_order = self.buy_orders[buy_index]
            sell_order = self.sell_orders[sell_index]

            if buy_order is None or sell_order is None:
                break

            if buy_order["size"] == 0:
                buy_index += 1
                continue

            if sell_order["size"] == 0:
                sell_index += 1
                continue

            if buy_order["trader"] == sell_order["trader"]:
                if side == "buy":
                    sell_index += 1
                    continue
                else:
                    buy_index += 1
                    continue

            # If the buy order is greater than or equal than the sell order
            if buy_order["price"] >= sell_order["price"]:
                # If buy order was submitted earlier than sell order, we'll use buy order's price
                if buy_order["time"] < sell_order["time"]:
                    trade_price = buy_order["price"]
                else:
                    # If sell order was submitted earlier than buy order, we'll use sell order's price
                    trade_price = sell_order["price"]

                trade_quantity = min(buy_order["size"], sell_order["size"])

                buy_trader = Trader.search_by_id(buy_order["trader"])
                sell_trader = Trader.search_by_id(sell_order["trader"])

                for i in range(trade_quantity):
                    buy_order["size"] -= 1
                    sell_order["size"] -= 1

                    result = buy_trader.transaction(
                        "buy", buy_order["ticker"], trade_price, trade_quantity)

                    if result is False:
                        trade_quantity = i
                        break

                    sell_trader.transaction(
                        "sell", sell_order["ticker"], trade_price, trade_quantity)

                timestamp = datetime.now()

                buy_trader.update_trade_history(
                    buy_order["ticker"], timestamp, "buy", trade_quantity, trade_price)
                sell_trader.update_trade_history(
                    sell_order["ticker"], timestamp, "sell", trade_quantity, trade_price)

                print("Order matched. Buyer of ID " + str(
                    buy_trader.id) + " bought " + str(trade_quantity) + " shares of stock " + self.ticker + " for "
                    "price " + str(trade_price) + " from seller of ID" + str(sell_trader.id) + ".")

                self.last.append({
                    "timestamp": timestamp,
                    "price": trade_price
                })

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
