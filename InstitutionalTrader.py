from Trader import Trader


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

    def analyze_recommendations(self, _0m, _1m, _2m, _3m):
        num_0m = self.__sum_analysts(_0m)
        num_1m = self.__sum_analysts(_1m)
        num_2m = self.__sum_analysts(_2m)
        num_3m = self.__sum_analysts(_3m)




