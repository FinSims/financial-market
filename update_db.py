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


# with open("constituents.csv") as dataFile:
#     reader = csv.DictReader(dataFile)
#
#     for row in reader:
#         ticker = row["Symbol"]
#         stock = yf.Ticker(ticker)
#
#         # _0m_dict, _1m_dict, _2m_dict, _3m_dict = get_recommendations(stock)
#
#         if "beta" in stock.info:
#             supabase.table("stock_list").update({
#                 "beta": stock.info["beta"]
#             }).eq("ticker", ticker).execute()
#         else:
#             supabase.table("stock_list").update({
#                 "beta": None
#             }).eq("ticker", ticker).execute()
