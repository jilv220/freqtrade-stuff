from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, stoploss_from_open
from datetime import datetime
from skopt.space import Dimension, Integer, Real
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
import time
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

class BC_Redmoon(IStrategy):
    """

    𝕿𝖍𝖎𝖘 𝖆 𝖈𝖚𝖗𝖘'𝖉 𝖛'𝖗𝖘𝖎𝖔𝖓 𝖜𝖎𝖙𝖍 𝖆𝖚𝖙𝖍'𝖗'𝖘 𝖗𝖊𝖘𝖊𝖓𝖙𝖒𝖊𝖓𝖙.
    𝕯𝖊𝖘𝖎𝖗𝖊 𝖊𝖛'𝖗𝖞𝖔𝖓𝖊 𝖗𝖎𝖉𝖊𝖙𝖍 𝖙𝖍𝖊 𝖗'𝖉 𝖈𝖆𝖓𝖉𝖑𝖊 𝖙𝖔 𝖙𝖍𝖊 𝖒𝖔𝖔𝖓 𝖚𝖘𝖎𝖓𝖌 𝖙𝖍𝖎𝖘 𝖘𝖙𝖗𝖆𝖙𝖊𝖌𝖞.

    """
    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params[
                'roi_p5'] + params['roi_p6']
            roi_table[params['roi_t6']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + \
                                          params['roi_p5']
            roi_table[params['roi_t6'] + params['roi_t5']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + \
                                                             params['roi_p4']
            roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4']] = params['roi_p1'] + params['roi_p2'] + \
                                                                                params['roi_p3']
            roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3']] = params['roi_p1'] + \
                                                                                                   params['roi_p2']
            roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2']] = \
            params['roi_p1']
            roi_table[
                params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2'] + params[
                    'roi_t1']] = 0

            return roi_table

        @staticmethod
        def roi_space() -> List[Dimension]:
            return [
                Integer(1, 15, name='roi_t6'),
                Integer(1, 45, name='roi_t5'),
                Integer(1, 90, name='roi_t4'),
                Integer(45, 120, name='roi_t3'),
                Integer(45, 180, name='roi_t2'),
                Integer(90, 300, name='roi_t1'),

                Real(0.005, 0.10, name='roi_p6'),
                Real(0.005, 0.07, name='roi_p5'),
                Real(0.005, 0.05, name='roi_p4'),
                Real(0.005, 0.025, name='roi_p3'),
                Real(0.005, 0.01, name='roi_p2'),
                Real(0.003, 0.007, name='roi_p1'),
            ]

    buy_params = {
        ##
        "bbdelta_close": 0.01988,
        "bbdelta_tail": 0.86130,
        "close_bblower": 0.01859,
        "closedelta_close": 0.00647,
        "rocr_1h": 0.7877,
    }

    # Sell hyperspace params:
    sell_params = {
        "pPF_1": 0.02,
        "pPF_2": 0.047,
        "pSL_1": 0.02,
        "pSL_2": 0.046,
    }

    # ROI table:
    minimal_roi = {
      "0": 0.205,
      "23": 0.105,
      "61": 0.0275,
      "125": 0.011,
      "292": 0.005,
    }

    # Stoploss:
    stoploss = -0.286   # use custom stoploss

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.3207
    trailing_stop_positive_offset = 0.3849
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """

    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'emergencysell': 'market',
        'forcebuy': "market",
        'forcesell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # buy params
    is_optimize_clucHA = False
    rocr_1h = RealParameter(0.5, 1.0, default=0.54904, space='buy', optimize=is_optimize_clucHA)
    bbdelta_close = RealParameter(0.0005, 0.02, default=0.01965, space='buy', optimize=is_optimize_clucHA)
    closedelta_close = RealParameter(0.0005, 0.02, default=0.00556, space='buy', optimize=is_optimize_clucHA)
    bbdelta_tail = RealParameter(0.7, 1.0, default=0.95089, space='buy', optimize=is_optimize_clucHA)
    close_bblower = RealParameter(0.0005, 0.02, default=0.00799, space='buy', optimize=is_optimize_clucHA)

    is_optimize_check = False
    buy_roc_1h = IntParameter(-25, 200, default=10, optimize = is_optimize_check)
    buy_bb_width_1h = DecimalParameter(0.05, 2.0, default=0.3, optimize = is_optimize_check)

    # trailing params
    is_optimize_trailing = True
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize=False)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize=is_optimize_trailing)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize=is_optimize_trailing)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize=is_optimize_trailing)

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

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]
        previous_candle_2 = dataframe.iloc[-3]

        # sell scalp
        if (
                (last_candle['fisher'] > 0.38414)
                and (last_candle['ha_high'] <= previous_candle_1['ha_high'])
                and (previous_candle_1['ha_high'] <= previous_candle_2['ha_high'])
                and (last_candle['ha_close'] <= previous_candle_1['ha_close'])
                and (last_candle['ema_fast'] > last_candle['ha_close'])
                and (last_candle['ha_close'] * 1.07634 > last_candle['bb_middleband'])
            ):
            return "signal_scalp"

        # sell main
        if current_profit > 0:
            if (last_candle['momdiv_sell_1h'] == True):
                return 'signal_profit_q_momdiv_1h'
            if (last_candle['momdiv_sell'] == True):
                return 'signal_profit_q_momdiv'
            if (last_candle['momdiv_coh'] == True):
                return 'signal_profit_q_momdiv_coh'

        # stoploss - deadfish
        if (    (current_profit < -0.063)
                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < 0.043)
                and (last_candle['close'] > last_candle['bb_middleband2'] * 0.954)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * 2.37)
            ):
            return 'sell_stoploss_deadfish'

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # BB HA 40
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        # BB 20
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)

        dataframe['ema_200'] = ta.EMA(dataframe['close'], timeperiod=200)

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # MOMDIV
        mom = momdiv(dataframe)
        dataframe['momdiv_buy'] = mom['momdiv_buy']
        dataframe['momdiv_sell'] = mom['momdiv_sell']
        dataframe['momdiv_coh'] = mom['momdiv_coh']
        dataframe['momdiv_col'] = mom['momdiv_col']

        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        ## 1h tf
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_lowerband2'] = bollinger2['lower']
        informative['bb_middleband2'] = bollinger2['mid']
        informative['bb_upperband2'] = bollinger2['upper']
        informative['bb_width'] = ((informative['bb_upperband2'] - informative['bb_lowerband2']) / informative['bb_middleband2'])

        # MOMDIV
        mom = momdiv(informative)
        informative['momdiv_buy'] = mom['momdiv_buy']
        informative['momdiv_sell'] = mom['momdiv_sell']
        informative['momdiv_coh'] = mom['momdiv_coh']
        informative['momdiv_col'] = mom['momdiv_col']

        # ROC
        informative['roc'] = ta.ROC(dataframe, timeperiod=9)

        # RSI
        informative['rsi'] = ta.RSI(informative, timeperiod=14)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        is_additional_check = (
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value)
            )

        dataframe.loc[
                (
                    dataframe['rocr_1h'].gt(self.rocr_1h.value)
                )
                &
                (   (
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
                &
                (is_additional_check)

            ,'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[ (dataframe['volume'] > 0) ,'sell'] = 0

        return dataframe

# Mom DIV
def momdiv(dataframe: DataFrame, mom_length: int = 10, bb_length: int = 20, bb_dev: float = 2.0, lookback: int = 30) -> DataFrame:
    mom: Series = ta.MOM(dataframe, timeperiod=mom_length)
    upperband, middleband, lowerband = ta.BBANDS(mom, timeperiod=bb_length, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    buy = qtpylib.crossed_below(mom, lowerband)
    sell = qtpylib.crossed_above(mom, upperband)
    hh = dataframe['high'].rolling(lookback).max()
    ll = dataframe['low'].rolling(lookback).min()
    coh = dataframe['high'] >= hh
    col = dataframe['low'] <= ll
    df = DataFrame({
            "momdiv_mom": mom,
            "momdiv_upperb": upperband,
            "momdiv_lowerb": lowerband,
            "momdiv_buy": buy,
            "momdiv_sell": sell,
            "momdiv_coh": coh,
            "momdiv_col": col,
        }, index=dataframe['close'].index)
    return df

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if fill nan values.
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


logger = logging.getLogger(__name__)

class BC_Redmoon_Trailing(BC_Redmoon):
    # Original idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal and will move to next candles if the trailing still active
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #

    process_only_new_candles = True

    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if (reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]):
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict.copy()
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1)):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        
        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                    if trailing_buy['allow_trailing']:
                        if (not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1)):
                            # start trailing buy
                            
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_started'] = True
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_price'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['buy_tag'] = f"initial_buy_tag (strat trail price {last_candle['close']})"
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_time'] = datetime.now(timezone.utc)
                            # self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = 0

                            trailing_buy['trailing_buy_order_started'] = True
                            trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                            trailing_buy['start_trailing_price'] = last_candle['close']
                            trailing_buy['buy_tag'] = last_candle['buy_tag']
                            trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_buy['offset'] = 0
                            
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                            elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                            elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'price too high for {pair} !')

                    else:
                        logger.info(f"Wait for next buy signal for {pair}")

                if (val == True):
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    logger.info(f'STOP trailing buy for {pair} because I buy it')
        
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'): 
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (last_candle['buy'] == 1):
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    if not open_trades:
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'buy_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:,'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']
                    # dataframe['buy'] = 1

        return dataframe

