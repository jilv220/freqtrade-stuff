import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta
import time
import logging

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter, IntParameter
from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
from technical.indicators import RMI

logger = logging.getLogger(__name__)

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class KamiNari(IStrategy):
    """
    Ultimate Scalper. Be fast and faster. Be like lightning (kaminari) and touch the shape of god.
    """
    
    #hypered params
    buy_params = {
        ##
        "max_slip": 0.73,
        ##
        "bbdelta_close": 0.01846,
        "bbdelta_tail": 0.98973,
        "close_bblower": 0.00785,
        "closedelta_close": 0.01009,
        "rocr_1h": 0.5411,
        ##
        "buy_hh_diff_48": 6.867,
        "buy_ll_diff_48": -12.884,
        ##
        "buy_ema_high": 1.031,
        "buy_ema_low": 0.962,
        "buy_ewo": 1.048,
        "buy_rsi": 29,
        "buy_rsi_fast": 36,
    }

    # Sell hyperspace params:
    sell_params = {
        "pPF_1": 0.011,
        "pPF_2": 0.064,
        "pSL_1": 0.011,
        "pSL_2": 0.062,

        # sell signal params
        "high_offset": 0.907,
        "high_offset_2": 1.211,
        "sell_bbmiddle_close": 0.97286,
        "sell_fisher": 0.48492,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.103,
        "3": 0.05,
        "5": 0.033,
        "61": 0.027,
        "125": 0.011,
        "292": 0.005,
    }

    # Stoploss:
    stoploss = -0.99  # use custom stoploss

    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    # buy params
    is_optimize_clucHA = False
    rocr_1h = RealParameter(0.5, 1.0, default=0.54904, space='buy', optimize = is_optimize_clucHA )
    bbdelta_close = RealParameter(0.0005, 0.02, default=0.01965, space='buy', optimize = is_optimize_clucHA )
    closedelta_close = RealParameter(0.0005, 0.02, default=0.00556, space='buy', optimize = is_optimize_clucHA )
    bbdelta_tail = RealParameter(0.7, 1.0, default=0.95089, space='buy', optimize = is_optimize_clucHA )
    close_bblower = RealParameter(0.0005, 0.02, default=0.00799, space='buy', optimize = is_optimize_clucHA )

    is_optimize_hh_ll = False
    buy_hh_diff_48 = DecimalParameter(0.0, 15, default=1.087 , optimize = is_optimize_hh_ll )
    buy_ll_diff_48 = DecimalParameter(-23, 40, default=1.087 , optimize = is_optimize_hh_ll )

    is_optimize_ewo = False
    buy_rsi_fast = IntParameter(35, 50, default=45, optimize = is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, optimize = is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 4, default=-5.585, optimize = is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942 , optimize = is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.9, 1.2, default=1.084 , optimize = is_optimize_ewo)

    ## slippage params
    is_optimize_slip = False
    max_slip = DecimalParameter(0.33, 0.80, default=0.33, decimals=3, optimize=is_optimize_slip , space='buy', load=True)

    # sell params
    is_optimize_sell = False
    sell_fisher = RealParameter(0.1, 0.5, default=0.38414, space='sell', optimize = is_optimize_sell)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=1.07634, space='sell', optimize = is_optimize_sell)
    high_offset          = DecimalParameter(0.90, 1.2, default=sell_params['high_offset'], space='sell', optimize = is_optimize_sell)
    high_offset_2        = DecimalParameter(0.90, 1.5, default=sell_params['high_offset_2'], space='sell', optimize = is_optimize_sell)

    # trailing params
    is_optimize_trailing = False
    pPF_1 = DecimalParameter(0.011, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)
    pSL_1 = DecimalParameter(0.011, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize = is_optimize_trailing)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    # come from BB_RPB_TSL
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        sl_profit = -0.99

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = -0.99

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    ## Confirm Entry
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        max_slip = self.max_slip.value

        if(len(dataframe) < 1):
            return False

        dataframe = dataframe.iloc[-1].squeeze()
        if ((rate > dataframe['close'])) :

            slippage = ( (rate / dataframe['close']) - 1 ) * 100

            if slippage < max_slip:
                return True
            else:
                return False

        return True

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]
        previous_candle_2 = dataframe.iloc[-3]

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate)

        # stoploss - deadfish
        if (
                (current_profit < -0.063)
                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < 0.043)
                and (last_candle['close'] > last_candle['bb_middleband2'] * 0.954)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * 2.37)
            ):
            return 'sell_stoploss_deadfish'

        # stoploss - pump
        if (last_candle['hl_pct_change_48_1h'] > 0.95):
            if (
                    (-0.04 > current_profit > -0.08)
                    and (max_profit < 0.005)
                    and (max_loss < 0.08)
                    and (last_candle['close'] < last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'])
                    and (last_candle['ema_vwma_osc_32'] < 0.0)
                    and (last_candle['ema_vwma_osc_64'] < 0.0)
                    and (last_candle['ema_vwma_osc_96'] < 0.0)
                    and (last_candle['cmf'] < -0.25)
                    and (last_candle['cmf_1h'] < -0.0)
            ):
                return 'sell_stoploss_p_48_1_1'
            elif (
                    (-0.04 > current_profit > -0.08)
                    and (max_profit < 0.01)
                    and (max_loss < 0.08)
                    and (last_candle['close'] < last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'])
                    and (last_candle['ema_vwma_osc_32'] < 0.0)
                    and (last_candle['ema_vwma_osc_64'] < 0.0)
                    and (last_candle['ema_vwma_osc_96'] < 0.0)
                    and (last_candle['cmf'] < -0.25)
                    and (last_candle['cmf_1h'] < -0.0)
            ):
                return 'sell_stoploss_p_48_1_2'

        if (last_candle['hl_pct_change_36_1h'] > 0.7):
            if (
                    (-0.04 > current_profit > -0.08)
                    and (max_loss < 0.08)
                    and (max_profit > (current_profit + 0.1))
                    and (last_candle['close'] < last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'])
                    and (last_candle['sma_200_dec_20_1h'])
                    and (last_candle['ema_vwma_osc_32'] < 0.0)
                    and (last_candle['ema_vwma_osc_64'] < 0.0)
                    and (last_candle['ema_vwma_osc_96'] < 0.0)
                    and (last_candle['cmf'] < -0.25)
                    and (last_candle['cmf_1h'] < -0.0)
            ):
                return 'sell_stoploss_p_36_1_1'

        if (last_candle['hl_pct_change_36_1h'] > 0.5):
            if (
                    (-0.05 > current_profit > -0.08)
                    and (max_loss < 0.08)
                    and (max_profit > (current_profit + 0.1))
                    and (last_candle['close'] < last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'])
                    and (last_candle['sma_200_dec_20_1h'])
                    and (last_candle['ema_vwma_osc_32'] < 0.0)
                    and (last_candle['ema_vwma_osc_64'] < 0.0)
                    and (last_candle['ema_vwma_osc_96'] < 0.0)
                    and (last_candle['cmf'] < -0.25)
                    and (last_candle['cmf_1h'] < -0.0)
                    and (last_candle['rsi'] < 40.0)
            ):
                return 'sell_stoploss_p_36_2_1'

        if (last_candle['hl_pct_change_24_1h'] > 0.6):
            if (
                    (-0.04 > current_profit > -0.08)
                    and (max_loss < 0.08)
                    and (last_candle['close'] < last_candle['ema_200'])
                    and (last_candle['sma_200_dec_20'])
                    and (last_candle['sma_200_dec_20_1h'])
                    and (last_candle['ema_vwma_osc_32'] < 0.0)
                    and (last_candle['ema_vwma_osc_64'] < 0.0)
                    and (last_candle['ema_vwma_osc_96'] < 0.0)
                    and (last_candle['cmf'] < -0.25)
                    and (last_candle['cmf_1h'] < -0.0)
            ):
                return 'sell_stoploss_p_24_1_1'

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        # BB 20
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        # BB 30
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)

        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_24'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['ema_26'] = ta.EMA(dataframe['close'], timeperiod=26)
        dataframe['ema_100'] = ta.EMA(dataframe['close'], timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe['close'], timeperiod=200)

        # SMA
        dataframe['sma_9'] = ta.SMA(dataframe['close'], timeperiod=9)
        dataframe['sma_28'] = ta.SMA(dataframe['close'], timeperiod=28)
        dataframe['sma_75'] = ta.SMA(dataframe['close'], timeperiod=75)
        dataframe['sma_200'] = ta.SMA(dataframe['close'], timeperiod=200)

        # HMA
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        # ROCR
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # hh48
        dataframe['hh_48'] = ta.MAX(dataframe['high'], 48)
        dataframe['hh_48_diff'] = (dataframe['hh_48'] - dataframe['close']) / dataframe['hh_48'] * 100

        # ll48
        dataframe['ll_48'] = ta.MIN(dataframe['low'], 48)
        dataframe['ll_48_diff'] = (dataframe['close'] - dataframe['ll_48']) / dataframe['ll_48'] * 100

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # RSI
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # sma dec 20
        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)

        # EMA of VWMA Oscillator
        dataframe['ema_vwma_osc_32'] = ema_vwma_osc(dataframe, 32)
        dataframe['ema_vwma_osc_64'] = ema_vwma_osc(dataframe, 64)
        dataframe['ema_vwma_osc_96'] = ema_vwma_osc(dataframe, 96)

        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, 50, 200)

        # CCI
        dataframe['cci_25'] = ta.CCI(dataframe, 25)

        # RMI
        dataframe['rmi_17'] = RMI(dataframe, length=17, mom=4)

        # SRSI
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # BB closedelta
        dataframe['bb_closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # T3 Average
        dataframe['T3'] = T3(dataframe)

        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)

        ### === 1h tf ===
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        informative['sma_200'] = ta.SMA(informative['close'], timeperiod=200)

        informative['hl_pct_change_48'] = range_percent_change(informative, 'HL', 48)
        informative['hl_pct_change_36'] = range_percent_change(informative, 'HL', 36)
        informative['hl_pct_change_24'] = range_percent_change(informative, 'HL', 24)
        informative['sma_200_dec_20'] = informative['sma_200'] < informative['sma_200'].shift(20)

        # T3 Average
        informative['T3'] = T3(informative)

        # BB 20
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_lowerband2'] = bollinger2['lower']
        informative['bb_middleband2'] = bollinger2['mid']
        informative['bb_upperband2'] = bollinger2['upper']

        # CMF
        informative['cmf'] = chaikin_money_flow(informative, 20)

        # CRSI (3, 2, 100)
        crsi_closechange = informative['close'] / informative['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        informative['crsi'] =  (ta.RSI(informative['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(informative['close'], 100)) / 3

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        is_bb = (
                    (dataframe['rmi_17'] < 49.0) &
                    (dataframe['cci_25'] < -116) &
                    (dataframe['srsi_fk'] < 32.0) &
                    (dataframe['bb_delta'] > 0.025) &
                    (dataframe['bb_width'] > 0.095) &
                    (dataframe['bb_closedelta'] > dataframe['close'] * 17.922 / 1000.0 ) &
                    (dataframe['close'] < (dataframe['bb_lowerband3'] * 0.999))
            )

        is_ewo = (
                    (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                    (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                    (dataframe['EWO'] < self.buy_ewo.value) &
                    (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                    (dataframe['rsi'] < self.buy_rsi.value)
            )

        is_ewo2 = (
                    (dataframe['rsi_fast'] < 45) &
                    (dataframe['close'] < dataframe['ema_8'] * 0.970) &
                    (dataframe['EWO'] > 4.179) &
                    (dataframe['close'] < dataframe['ema_16'] * 1.087) &
                    (dataframe['rsi'] < 35)
            )

        is_ClucHA = (
                (
                    (dataframe['lower'].shift().gt(0)) &
                    (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value)) &
                    (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value)) &
                    (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value)) &
                    (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                    (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
                )
                |
                (
                    (dataframe['ha_close'] < dataframe['ema_slow']) &
                    (dataframe['ha_close'] < self.close_bblower.value * dataframe['bb_lowerband'])
                )
            )

        is_cofi = (
                    (dataframe['open'] < dataframe['ema_8'] * 1.147) &
                    (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                    (dataframe['fastk'] < 39) &
                    (dataframe['fastd'] < 28) &
                    (dataframe['adx'] > 13) &
                    (dataframe['EWO'] > 8.594) &
                    (dataframe['cti'] < -0.892) &
                    (dataframe['r_14'] < -85.016)
            )

        is_gumbo = (
                    (dataframe['EWO'] < -9.442) &
                    (dataframe['bb_middleband2_1h'] >= dataframe['T3_1h']) &
                    (dataframe['T3'] <= dataframe['ema_8'] * 1.121) &
                    (dataframe['cti'] < -0.374) &
                    (dataframe['r_14'] < -51.971)
            )

        is_nfi_38 = (
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.98) &
                (dataframe['EWO'] < -4.4) &
                (dataframe['cti'] < -0.95) &
                (dataframe['r_14'] < -97) &
                (dataframe['crsi_1h'] > 0.5)
            )

        is_r_deadfish = (
                (dataframe['ema_100'] < dataframe['ema_200'] * 1.054) &
                (dataframe['bb_width'] > 0.34) &
                (dataframe['close'] < dataframe['bb_middleband2'] * 1.014) &
                (dataframe['volume_mean_12'] > dataframe['volume_mean_24'] * 1.6) &
                (dataframe['cti'] < -0.115) &
                (dataframe['r_14'] < -44.34)
            )

        dataframe.loc[

            (
                (dataframe['rocr_1h'].gt(self.rocr_1h.value)) &
                (is_ClucHA) &
                (dataframe['hh_48_diff'] > self.buy_hh_diff_48.value) &
                (dataframe['ll_48_diff'] > self.buy_ll_diff_48.value)
            )
            |
            (is_ewo2)
            |
            (is_ewo)
            |
            (is_bb)
            |
            (is_cofi)
            |
            (is_gumbo)
            |
            (is_nfi_38)
            |
            (is_r_deadfish)

        ,'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (   (
                    (dataframe['fisher'] > self.sell_fisher.value) &
                    (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
                    (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
                    (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
                    (dataframe['ema_fast'] > dataframe['ha_close']) &
                    ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband'])
                )
                |
                (
                    (dataframe['close'] > dataframe['sma_9']) &
                    (dataframe['close'] > (dataframe['ema_24'] * self.high_offset_2.value)) &
                    (dataframe['rsi'] > 50) &
                    (dataframe['rsi_fast'] > dataframe['rsi_slow'])
                )
                |
                (
                    (dataframe['sma_9'] > (dataframe['sma_9'].shift(1) + dataframe['sma_9'].shift(1) * 0.005 )) &
                    (dataframe['close'] < dataframe['hma_50']) &
                    (dataframe['close'] > (dataframe['ema_24'] * self.high_offset.value)) &
                    (dataframe['rsi_fast'] > dataframe['rsi_slow'])
                )
            )
            &
            (dataframe['volume'] > 0)

        ,'sell'] = 1

        return dataframe

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    vwma = vwma.fillna(0, inplace=True)
    return vwma

# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

# Eillot Wave Oscillator
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx
