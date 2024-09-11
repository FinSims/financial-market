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
        # Make this choose a stock with the hihgest beta and volatility
