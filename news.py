import random


class News:
    def __init__(self, stocks):
        self.stocks = stocks
        self.sentiment = 0

    def spit_news(self):
        chosen_stock = random.choice(self.stocks)
        self.sentiment = self.generate_sentiment()
        return {
            "stock": chosen_stock,
            "sentiment": self.sentiment,
            "headline": self.generate_headline(chosen_stock, self.sentiment)
        }

    def generate_sentiment(self):
        probabilities = [0.04, 0.20, 0.50, 0.19, 0.07]
        sentiment_levels = [1, 2, 3, 4, 5]
        return random.choices(sentiment_levels, probabilities)[0]

    def generate_headline(self, stock, sentiment):
        headlines = {
            1: f"{stock} faces major setbacks: Investors brace for impact",
            2: f"Challenges ahead for {stock}: Cautious optimism in the market",
            3: f"Steady performance for {stock}: Market remains neutral",
            4: f"{stock} shows promising growth: Analysts expect positive trends",
            5: f"{stock} soars to new heights: Investors celebrate remarkable gains"
        }
        return headlines.get(sentiment, "Market update for " + stock)
