from .TradingStrategy import TradingStrategy
from ..utilities.Security import Security
from ..services.SupabaseClient import SupabaseClient
from supabase import Client
import math
import yfinance


# Generates trade signal on a min and max percentile based on a stock rating based on analyst recommendations
class DeepValueStrategy(TradingStrategy):
    def __init__(self, pe_threshold=20, pb_threshold=1.0, debt_to_equity_threshold=1.0):
        """
        Initialize the Deep Value Strategy with specific thresholds.
        """
        self.pe_threshold = pe_threshold
        self.pb_threshold = pb_threshold
        self.de_threshold = debt_to_equity_threshold

    def execute(self, stock_ticker, balance, create_limit_order, time):
        """
        Execute the Deep Value Strategy for a given stock.
        """
        stock = Security.get_instance(stock_ticker)
        pe_ratio = stock.calculate_pe_ratio()
        pb_ratio = stock.calculate_pb_ratio()
        de_ratio = stock.calculate_de_ratio()

        if pe_ratio is not None and pb_ratio is not None and de_ratio is not None:
            if (pe_ratio <= self.pe_threshold and
                    pb_ratio <= self.pb_threshold and
                    de_ratio <= self.de_threshold):

                quantity = math.floor((balance * .05) /
                                      stock.last[0]["price"])

                if quantity > 0:
                    print("\nSubmitted buy order for stock: " +
                          stock_ticker + " at " + time + " using the Deep Value Strategy")
                    create_limit_order(stock_ticker, "buy", stock.ask, quantity)
        # stock = Security.get_instance(stock_ticker)
        # quantity = math.floor((balance * .05) /
        #                       stock.last[0]["price"])
        # print("\nSubmitted short sell order for stock: " +
        #       stock_ticker + " at " + time)
        # create_limit_order(stock_ticker, "sell", stock.bid, quantity)
