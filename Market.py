import random

import simpy
import datetime
from utilities.Trader import Trader
import uuid
from services.SupabaseClient import SupabaseClient
from supabase import Client
from InstitutionalTrader import InstitutionalTrader
import random
from utilities.Security import Security
import yfinance as yf


class Market:
    def __init__(self, total_traders, percent_inst, inst_min, inst_max):
        self.simulation_id = uuid.uuid4()
        self.total_traders = total_traders
        self.percent_inst = percent_inst
        self.inst_min = inst_min
        self.inst_max = inst_max
        self.supabase: Client = SupabaseClient.get_instance()

    @staticmethod
    def _format_time(env_time):
        market_open = datetime.time(9, 30)
        minutes = int(env_time)
        market_time = (datetime.datetime.combine(datetime.date.today(), market_open) +
                       datetime.timedelta(minutes=minutes))
        return market_time.strftime("%I:%M %p")

    # Creates the IPO Trader, which will be used to provide liquidity for the start of the simulation
    @staticmethod
    def __init_ipo():
        ipo_trader = Trader(True, 100000000000)
        return ipo_trader.id

    # Creates all the institutional traders and add them to the database
    def __init_inst(self):
        # Clear all entries of existing traders to set up a new simulation
        self.supabase.table("traders").delete().eq(
            "type", "institutional").execute()

        inst_traders = int(self.percent_inst * self.total_traders)

        for i in range(inst_traders):
            balance = random.uniform(self.inst_min, self.inst_max)
            trader = InstitutionalTrader(balance)

            self.supabase.table("traders").insert({
                "simulation_id": str(self.simulation_id),
                "trader_id": str(trader.id),
                "type": "institutional",
                "balance": balance,
                "canShort": True
            }).execute()

    def __process_day(self, _env, trader_response, stock_response, ipo_id):
        while True:
            time = self._format_time(_env.now)

            # Chooses a random institutional trader
            random_trader = random.choice(trader_response.data)
            random_stock = random.choice(
                stock_response.data)  # Chooses a random stock

            # If the stock is not initialized yet, we initialize it by creating a new Security object
            if Security.get_instance(random_stock["ticker"]) is None:
                Security(random_stock["ticker"])
                yf_stock = yf.Ticker(random_stock["ticker"])
                ipo_trader = Trader.search_by_id(ipo_id)
                ipo_trader.create_limit_order(random_stock["ticker"], "sell",
                                              yf_stock.info["ask"],
                                              yf_stock.info[
                                                  "sharesOutstanding"])

            trader = InstitutionalTrader.search_by_id(
                uuid.UUID(random_trader["trader_id"]))
            # Buys a stock if percentile is over 75, short sells stock if percentile is below 20
            trader.generate_trade_signal(random_stock["ticker"], 75, 20, time)
            yield _env.timeout(1)  # Advance by one minute

    def open_market(self):
        # Init IPO trader that creates liquidity in the market initially
        ipo_id = self.__init_ipo()
        self.__init_inst()  # Creates all institutional traders

        env = simpy.Environment()  # Create trading environment

        trader_response = self.supabase.table("traders").select(
            "*").eq("type", "institutional").execute()
        stock_response = self.supabase.table(
            "stock_list").select("*").execute()

        # Run the __process_day method within our environment
        env.process(self.__process_day(
            env, trader_response, stock_response, ipo_id))
        # Run the simulation until it reaches 390
        env.run(until=390)  # 390 minutes in one trading day


# Creates the market object with the minimum balance for institutional traders being 9999999 and max being 100000000
market = Market(100, 0.1, 9999999, 100000000)

market.open_market()
