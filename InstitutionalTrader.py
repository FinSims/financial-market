import math
from Trader import Trader
from SupabaseClient import SupabaseClient
from supabase import Client
from Security import Security


class InstitutionalTrader(Trader):
    def __init__(self, balance):
        super().__init__(True, balance)

    @staticmethod
    def __get_stock_rating(month_dict):
        recommendation_values = {
            'strongBuy': 5,
            'buy': 4,
            'hold': 3,
            'sell': 2,
            'strongSell': 1,
        }

        total_score = 0
        total_count = 0

        # Loop through the recommendations and calculate the weighted average
        for recommendation, count in month.items():
            total_score += recommendation_values[recommendation] * count
            total_count += count

        # Calculate the weighted average rating
        if total_count > 0:
            weighted_average_rating = total_score / total_count
        else:
            weighted_average_rating = 0

        # Print the weighted average rating
        print("The weighted average rating is:", weighted_average_rating)

    def generate_trade_signal(self, stock_ticker, minimum_percentile):
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
        weighted_percentile = (total_tickers - other_rank) / total_tickers * 100

        overall_percentile = (percentile + weighted_percentile) / 2

        if overall_percentile >= minimum_percentile:
            stock = Security.get_instance(stock_ticker)
            quantity = math.floor((self.balance * .05) / stock.last[0]["price"])
            super().create_market_order(stock_ticker, "buy", quantity)

        return overall_percentile


# trader = InstitutionalTrader(3)
# response = trader.generate_trade_signal("KVUE")
#
# print(response)
