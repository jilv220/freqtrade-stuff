import logging
from numpy.lib import math
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib


class SuperTrendPure(IStrategy):
   
    # ROI table:
    minimal_roi = {
        "0": 0.087,
        "372": 0.058,
        "861": 0.029,
        "2221": 0
    }
    # Stoploss:
    stoploss = -0.265

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.144
    trailing_only_offset_is_reached = False

    timeframe = '1h'

    startup_candle_count = 50
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
       
        supertrend = self.supertrend(dataframe, 2, 8)
        dataframe['st'] = supertrend['ST']
        dataframe['stx'] = supertrend['STX']
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
               (qtpylib.crossed_above(dataframe['close'], dataframe['st'])) &
               (dataframe['volume'].gt(0))
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
               (qtpylib.crossed_below(dataframe['close'], dataframe['st'])) &
               (dataframe['volume'].gt(0))
            ),
            'sell'] = 1

        return dataframe

    """
        Supertrend Indicator; adapted for freqtrade
        from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """
    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00
        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST' : df[st],
            'STX' : df[stx]
        })
