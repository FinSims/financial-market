from .TradingStrategy import TradingStrategy
from ..utilities.Security import Security
from ..services.SupabaseClient import SupabaseClient
from supabase import Client
import math


class AnalystInformedStrategy(TradingStrategy):
    def __init__(self, min_percentile, max_percentile):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def execute(self, stock_ticker, balance, create_limit_order, time):
        """
        Execute the Analyst Informed Strategy for a given stock, using analyst recommendations to buy or short a
        given stock.
        """
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

        if overall_percentile >= self.min_percentile:
            stock = Security.get_instance(stock_ticker)
            quantity = math.floor((balance * .05) /
                                  stock.last[0]["price"])
            print("\nSubmitted buy order for stock: " +
                  stock_ticker + " at " + time + " using the Analyst Informed Strategy")
            create_limit_order(stock_ticker, "buy", stock.ask, quantity)

        if overall_percentile < self.max_percentile:
            stock = Security.get_instance(stock_ticker)
            quantity = math.floor((balance * .05) /
                                  stock.last[0]["price"])
            print("\nSubmitted short sell order for stock: " +
                  stock_ticker + " at " + time + " using the Analyst Informed Strategy")
            create_limit_order(stock_ticker, "sell", stock.bid, quantity)
