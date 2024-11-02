from abc import ABC, abstractmethod


class TradingStrategy(ABC):
    @abstractmethod
    def execute(self, stock_ticker, balance, create_limit_order, time):
        pass