from Trader import Trader
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
import numpy as np
from QuantPredictor import QuantPredictor

# LSTM Will give price prediciton on the next day
# Based on that hey buy or sell
# Parameters -- reinforcing, does either adjust parameter of model or adjust trust in the model


class quantTrader(Trader):
    def __init__(self, balance):
        super().__init__(True, balance)

    def getStockRec(self, stock):
        predictor = QuantPredictor(lookback=7, hidden_size=16,
                                   num_stacked_layers=2, batch_size=16, learning_rate=0.003, num_epochs=30)
        result_df = predictor.predict_stock(str(stock))
        return result_df
