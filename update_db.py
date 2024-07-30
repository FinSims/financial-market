# Don't run this file

import yfinance as yf
import csv
from dotenv import load_dotenv
import os
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def get_recommendations(stock):
    data = stock.recommendations.to_dict()

    periods = data['period']
    strong_buy = data['strongBuy']
    buy = data['buy']
    hold = data['hold']
    sell = data['sell']
    strong_sell = data['strongSell']

    period_dicts = {}

    for i in periods:
        period_dicts[periods[i]] = {
            'strongBuy': strong_buy[i],
            'buy': buy[i],
            'hold': hold[i],
            'sell': sell[i],
            'strongSell': strong_sell[i]
        }

    _0m_dict = period_dicts["0m"]
    _1m_dict = period_dicts["-1m"]
    _2m_dict = period_dicts["-2m"]
    _3m_dict = period_dicts["-3m"]

    return _0m_dict, _1m_dict, _2m_dict, _3m_dict


def get_stock_rating(month_dict):
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
    for recommendation, count in month_dict.items():
        total_score += recommendation_values[recommendation] * count
        total_count += count

    # Calculate the weighted average rating
    if total_count > 0:
        weighted_average_rating = total_score / total_count
    else:
        weighted_average_rating = None

    # Print the weighted average rating
    return weighted_average_rating


# pltr = yf.Ticker("PLTR")
# print(pltr.history(period="1d", start="2021-01-01", end="2021-12-31"))


with open("constituents.csv") as dataFile:
    reader = csv.DictReader(dataFile)

    for row in reader:
        ticker = row["Symbol"]
        stock = yf.Ticker(ticker)

        _0m_dict, _1m_dict, _2m_dict, _3m_dict = get_recommendations(stock)
        rating_0m = get_stock_rating(_0m_dict)
        rating_1m = get_stock_rating(_1m_dict)
        rating_2m = get_stock_rating(_2m_dict)
        rating_3m = get_stock_rating(_3m_dict)

        # if ticker == "AMCR":
        #     print(_2m_dict)
        #     break

        ratings = [rating_0m, rating_1m, rating_2m, rating_3m]
        weights = [0.4, 0.3, 0.2, 0.1]

        # Filter out the None values
        valid_ratings_weights = [(r, w) for r, w in zip(ratings, weights) if r is not None]

        if not valid_ratings_weights:  # If all ratings are None
            print(ticker)
            overall_stock_rating = None
        else:
            weighted_sum = sum(r * w for r, w in valid_ratings_weights)
            total_weight = sum(w for _, w in valid_ratings_weights)
            overall_stock_rating = weighted_sum / total_weight

        supabase.table("stock_list").update({
            "overall_stock_rating": overall_stock_rating
        }).eq("ticker", ticker).execute()
