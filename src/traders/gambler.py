import yfinance as yf
from datetime import datetime, time
import random
import pandas as pd
from supabase import create_client, Client
import sys
import os
from ..services.SupabaseClient import SupabaseClient


class Gambler:
    def __init__(self) -> None:

        self.supabase: Client = SupabaseClient.get_instance()

    def get_stocks(self):
        stocks = self.supabase.table("stock_list").select("*").execute()
        return stocks

    def choose_stock(self):
        stocks = self.get_stocks()
        stocks = stocks.data
        # CHooses tock if beta over 1.3 which is considered high
        high_beta_stocks = [
            stock for stock in stocks if stock.get('beta') is not None and stock.get('beta', 0) > 1.3
        ]

        high_beta_stocks.sort(key=lambda x: x.get('volume', 0), reverse=True)

        high_beta_stocks.sort(key=lambda x: x.get('beta', 0), reverse=True)
        chosen_stock = random.choice(high_beta_stocks)

        return chosen_stock
