import random
from SupabaseClient import SupabaseClient

client = SupabaseClient.get_instance()


class News:
    def __init__(self):
        self.sentiment = 0

    # def spit_news(self):
    #     chosen_stock = random.choice(self.stocks)
    #     self.sentiment = self.generate_sentiment()
    #     return {
    #         "stock": chosen_stock,
    #         "sentiment": self.sentiment,
    #         "headline": self.generate_headline(chosen_stock, self.sentiment)
    #     }

    def generate_sentiment(self):
        probabilities = [0.04, 0.20, 0.50, 0.19, 0.07]
        sentiment_levels = [1, 2, 3, 4, 5]
        return random.choices(sentiment_levels, probabilities)[0]

    def generate_sentiment_stock(self, stock):
        stock = client.table('stock_list').select(
            '*').eq('ticker', stock).execute()
        sentiment_list = stock.data

        for sentiment in sentiment_list:
            overall_stock_rating = sentiment.get('overall_stock_rating', 3)
            rounded_rating = round(overall_stock_rating)

            weights = [1, 1, 1, 1, 1]
            if 1 <= rounded_rating <= 5:
                weights[rounded_rating - 1] = 5

            generated_sentiment = random.choices(
                range(1, 6), weights=weights, k=1)[0]
            return generated_sentiment

    def generate_headline(self, stock, sentiment):
        headlines = {
            1: f"{stock} gets set to fall: Investors brace for PAINAL",
            2: f"{stock} might have some challenges ahead: Investors are edging, but might not make it during NNN",
            3: f"{stock} is flat: Asian Woman council has inducted it into hall of fame",
            4: f"{stock} shows promising growth from chode to slightly below average: Analysts expect penis enlargement pills to kick in soon",
            5: f"{stock} reaches CLIMAX: All the investors are getting  pounded by their wives"
        }

        return headlines.get(sentiment, "Market update for " + stock)


# 1-5 will influence wheter or not they get a 1-5 headline. If null just pick random, if rating is between 1-5, give it lik ea 66% percent change of the same rating


# Exmple Usage
'''
news = News()
print(news.generate_headline("AAPL", news.generate_sentiment_stock('AAPL')))

'''
