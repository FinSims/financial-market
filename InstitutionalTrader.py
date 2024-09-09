from utilities.Trader import Trader
from utilities.Security import Security
from services.SupabaseClient import SupabaseClient
from supabase import Client
import math


class InstitutionalTrader(Trader):
    def __init__(self, balance):
        super().__init__(True, balance)

    def generate_trade_signal(self, stock_ticker, min_percentile, max_percentile, time):
        supabase_instance: Client = SupabaseClient.get_instance()

        ordered_tickers = supabase_instance.table("stock_list").select("ticker").order(
            "overall_stock_rating", desc=True).execute()

        weighted_ordered_tickers = supabase_instance.table("stock_list").select("ticker").order(
            "weighted_overall_stock_rating", desc=True).execute()

        total_tickers = len(ordered_tickers.data)
        rank = 0

        for ticker in ordered_tickers.data:
            rank += 1
            if ticker["ticker"] == stock_ticker:
                break

        other_rank = 0

        for ticker in weighted_ordered_tickers.data:
            other_rank += 1
            if ticker["ticker"] == stock_ticker:
                break

        percentile = (total_tickers - rank) / total_tickers * 100
        weighted_percentile = (
            total_tickers - other_rank) / total_tickers * 100

        overall_percentile = (percentile + weighted_percentile) / 2

        if overall_percentile >= min_percentile:
            stock = Security.get_instance(stock_ticker)
            quantity = math.floor((self.balance * .05) /
                                  stock.last[0]["price"])
            print("\nSubmitted buy order for stock: " +
                  stock_ticker + " at " + time)
            super().create_limit_order(stock_ticker, "buy", stock.ask, quantity)

        if overall_percentile < max_percentile:
            stock = Security.get_instance(stock_ticker)
            quantity = math.floor((self.balance * .05) /
                                  stock.last[0]["price"])
            print("\nSubmitted short sell order for stock: " +
                  stock_ticker + " at " + time)
            super().create_limit_order(stock_ticker, "sell", stock.bid, quantity)


trader = InstitutionalTrader(1000000)
trader.generate_trade_signal("AAPL", 75, 20, "9:30")
