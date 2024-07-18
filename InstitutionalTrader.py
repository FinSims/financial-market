from Trader import Trader
from SupabaseClient import SupabaseClient
from supabase import Client


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

    def analyze_recommendations(self):
        supabase_instance: Client = SupabaseClient.get_instance()

        response = supabase_instance.table("stock_list").select("*").gte("0m_rating", 0).order("0m_rating",
                                                                                                  desc=True).limit(
            20).order("weighted_0m_rating").execute()

        return response


trader = InstitutionalTrader(3)
response = trader.analyze_recommendations()

for row in response.data:
    print(row["ticker"], row["0m_rating"], row["weighted_0m_rating"], row["-1m_rating"],
          row["weighted_-1m_rating"])
