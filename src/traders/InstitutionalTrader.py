from ..utilities.Trader import Trader
from ..strategies.TradingStrategy import TradingStrategy


class InstitutionalTrader(Trader):
    def __init__(self, balance, strategy: TradingStrategy):
        super().__init__(True, balance)
        self.strategy = strategy

    def execute_trade(self, stock_ticker, time):
        self.strategy.execute(stock_ticker, self.balance, super().create_limit_order, time)

    def set_strategy(self, strategy: TradingStrategy):
        self.strategy = strategy

