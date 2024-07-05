from typing import Optional, Literal
from datetime import datetime
import uuid


class Trader:
    _instances = []

    def __init__(self, shorting, balance=0):
        self.id = uuid.uuid4()
        self.balance = balance
        self.init_balance = balance
        self.portfolio = []
        self.trade_history = []
        # Whether the trader can short sell stocks
        self.shorting = shorting
        Trader._instances.append(self)

    @classmethod
    def search_by_id(cls, class_id) -> Optional['Trader']:
        for instance in cls._instances:
            if instance.id == class_id:
                return instance
        return None

    # Finds the given stock in the trader's portfolio
    def find_stock(self, ticker):
        portfolio_stock = next((stock for stock in self.portfolio if stock["ticker"] == ticker), None)
        return portfolio_stock

    def update_trade_history(self, ticker, timestamp, side, quantity, price):
        self.trade_history.append({
            "time": timestamp,
            "ticker": ticker,
            "type": side,
            "price": price,
            "quantity": quantity
        })

    def display_trade_history(self, ascending):
        if ascending:
            self.trade_history.sort(key=lambda x: x["time"])
        else:
            self.trade_history.sort(key=lambda x: x["time"], reverse=True)

        return self.trade_history

    def create_market_order(self, ticker, side, quantity):
        from Security import Security
        instrument = Security.get_instance(ticker)

        if instrument is None:
            instrument = Security(ticker)
        instrument.update_bid_ask()

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
            portfolio_stock = self.find_stock(ticker)

            if portfolio_stock is not None and portfolio_stock["size"] < quantity and self.shorting is False:
                print("Error: Order not accepted. Not enough stock to sell. Short selling is disabled.")
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
        from Security import Security
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
            portfolio_stock = self.find_stock(ticker)

            if (portfolio_stock is None and self.shorting is False) or (
                    portfolio_stock is not None and portfolio_stock["size"] < quantity and self.shorting is False):
                print("Error: Order not accepted. Not enough stock to sell. Short selling is disabled.")
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

        instrument.execute_limit_order(side)

    def transaction(self, side, ticker, price, quantity):
        # stock = Security.get_instance(ticker)
        # Finds if the trader already owns the stock in their portfolio
        portfolio_stock = self.find_stock(ticker)

        if side == "buy":
            if self.balance >= price:
                if portfolio_stock is None:
                    self.portfolio.append({
                        "ticker": ticker,
                        "size": 1,
                        "avg_price": price
                    })

                    self.balance -= price
                else:
                    # If we already own shares of it
                    if portfolio_stock["size"] > 0:
                        new_avg_price = ((portfolio_stock["avg_price"] * portfolio_stock["size"]) + (
                                    price * 1)) / (
                                                portfolio_stock["size"] + 1)
                        portfolio_stock["avg_price"] = new_avg_price
                        portfolio_stock["size"] += 1
                        self.balance -= price
                    else:
                        # If we are covering back a short as quantity is in negatives
                        portfolio_stock["size"] += 1
                        self.balance -= price
            else:
                print("Error: Order not accepted. Not enough funds to buy all " + str(quantity) + " shares of stock " +
                      ticker)
                return False
        else:
            if portfolio_stock is not None:
                if self.shorting is True and portfolio_stock["size"] < 0:
                    # If we are already shorting some shares, we'll calculate new average price
                    new_avg_price = ((portfolio_stock["avg_price"] * abs(portfolio_stock["size"])) + (price * 1)) / (
                            abs(portfolio_stock["size"]) + 1)
                    portfolio_stock["avg_price"] = new_avg_price
                    portfolio_stock["size"] -= 1
                    self.balance += price
                elif portfolio_stock["size"] >= 1:
                    portfolio_stock["size"] -= 1
                    self.balance += price
                    # if portfolio_stock["size"] == 0:
                    #     portfolio_stock["avg_price"] = 0
                elif self.shorting is False and portfolio_stock["size"] < 1:
                    # If we are trying to sell more than we already own...
                    print("Error: Order not accepted. Short selling is disabled.")
                    return False
            else:  # We don't already own the shares
                if self.shorting is True:
                    self.portfolio.append({
                        "ticker": ticker,
                        "size": -1,
                        "avg_price": price
                    })
                    self.balance += price
                else:
                    print("Error: Order not accepted. No shares of " + ticker + " already owned in portfolio.")
                    return False

    def percent_change(self):
        # check if initial balance is 0
        init_balance = float(self.init_balance) if self.init_balance != 0 else .01
        balance = self.balance

        if balance == init_balance:
            return 0
        else:
            return ((balance - init_balance) / init_balance) * 100