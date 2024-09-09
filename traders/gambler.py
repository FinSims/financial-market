import yfinance as yf
from datetime import datetime, time
import random
import pandas as pd
from supabase import create_client, Client
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Gambler:
    def __init__(self) -> None:
        from services.SupabaseClient import SupabaseClient

        self.supabase: Client = SupabaseClient.get_instance()

    def get_stocks(self):
        stocks = self.supabase.table("stock_list").select("*").execute()
        return stocks


g = Gambler()
print(g.get_stocks())
