class Security:
    def __init__(self, ticker):
        self.ticker = ticker
        self.order_book = []
        self.trade_history = []
        self.bid = 0
        self.ask = 0
        self.last = 0

    def create_limit_order(self, side, price, quantity):
        # Finds first order in which side is same and price is same, so we can add on the quantity if it exists
        matching_order_index = next(
            (i for i, order in enumerate(self.order_book) if order["side"] == side and order["price"] == price), None)

        if matching_order_index:
            self.order_book[matching_order_index]["size"] += quantity
        else:
            self.order_book.append({
                "ticker": self.ticker,
                "side": side,
                "price": price,
                "size": quantity,
                "type": "limit"
            })

        self.match_orders()
    def create_market_order(self, side, quantity):
        buy_orders, sell_orders = self.__update_bid_ask()

        if side == "buy":
            self.order_book.append({
                "ticker": self.ticker,
                "side": side,
                "price": sell_orders[0]["price"],
                "size": quantity,
                "type": "market"
            })
        else:
            self.order_book.append({
                "ticker": self.ticker,
                "side": side,
                "price": buy_orders[0]["price"],
                "size": quantity,
                "type": "market"
            })

        self.match_orders()

    def __update_bid_ask(self):
        buy_orders = [order for order in self.order_book if order["side"] == "buy"]
        buy_orders.sort(key=lambda order: order["price"], reverse=True)  # sort from highest to lowest
        sell_orders = [order for order in self.order_book if order["side"] == "sell"]
        sell_orders.sort(key=lambda order: order["price"], reverse=False)  # sort from lowest to highest

        if len(buy_orders) > 0:
            self.bid = max(buy_orders, key=lambda x: x['price'])
        else:
            self.bid = 0

        if len(sell_orders) > 0:
            self.ask = min(sell_orders, key=lambda x: x['price'])
        else:
            self.ask = 0

        return buy_orders, sell_orders

    def match_orders(self):
        # buy_orders and sell_orders are sorted appropriately
        buy_orders, sell_orders = self.__update_bid_ask()

        buy_index = 0
        sell_index = 0

        while buy_index < len(buy_orders) and sell_index < len(sell_orders):
            buy_order = buy_orders[buy_index]
            sell_order = sell_orders[sell_index]

            if buy_order is None or sell_order is None:
                break

            # If we have a buy and sell order, and the buy order is greater than or equal than the sell order
            if buy_order["type"] == "limit" and buy_order["price"] >= sell_order["price"]:
                trade_price = sell_order["price"]
                trade_quantity = min(buy_order["size"], sell_order["size"])
                self.trade_history.append({"price": trade_price, "quantity": trade_quantity})
                buy_order["size"] -= trade_quantity
                sell_order["size"] -= trade_quantity

                # If this buy order is overpaying, and they still have shares left, there is a possibility they can
                # get the rest of their shares filled, so we'll move up the sellers list
                if sell_order["size"] == 0 and buy_order["size"] > 0 and buy_order["price"] > sell_order["price"]:
                    # Note that this may not be filled if it does not meet the limit criteria of the buyer
                    sell_index += 1
                else:
                    break
            elif buy_order["type"] == "market":
                trade_price = sell_order["price"]
                trade_quantity = min(buy_order["size"], sell_order["size"])
                self.trade_history.append({"price": trade_price, "quantity": trade_quantity})
                buy_order["size"] -= trade_quantity
                sell_order["size"] -= trade_quantity

                if sell_order["size"] == 0 and buy_order["size"] > 0:
                    sell_index += 1
                else:
                    break
            else:
                # If the bids and asks cannot match up, then the market cannot execute an order
                break



my_stock = Security("AAPL")
# print(my_stock.order_book)
my_stock.create_limit_order("sell", 100.0, 5)
# print(my_stock.order_book)
my_stock.create_limit_order("sell", 101.0, 5)
# print(my_stock.order_book)
my_stock.create_limit_order("sell", 102.0, 5)
# print(my_stock.order_book)
my_stock.create_limit_order("sell", 103.0, 5)
# print(my_stock.order_book)
my_stock.create_limit_order("sell", 104.0, 5)
# print(my_stock.order_book)
my_stock.create_limit_order("sell", 105.0, 5)
# print(my_stock.order_book)
# my_stock.create_limit_order("sell", 99.94, 10)
# print(my_stock.order_book)
# my_stock.create_limit_order("sell", 99.99, 2)
# print(my_stock.order_book)
my_stock.create_market_order("buy", 50)
print(my_stock.trade_history)
