from ..utilities.Trader import Trader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .QuantPredictor import QuantPredictor
# LSTM Will give price prediciton on the next day
# Based on that hey buy or sell
# Parameters -- reinforcing, does either adjust parameter of model or adjust trust in the model


class QuantTrader(Trader):
    def __init__(self, balance):
        super().__init__(True, balance)

    def get_stock_recc(self, stock):
        predictor = QuantPredictor(lookback=5, hidden_size=64,
                                   num_stacked_layers=4, batch_size=16, learning_rate=0.003, num_epochs=30)
        result_df = predictor.predict_stock(str(stock))
        return result_df
