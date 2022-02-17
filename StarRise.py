# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import talib.abstract as ta
import numpy as np

from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open

import math
import logging

logger = logging.getLogger(__name__)

# --------------------------------
def top_percent_change(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


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

class StarRise(IStrategy):
    """

    Designed to use with StarRise DCA settings

    TTP: 1.1%(0.2%), BO: 38.0 USDT, SO: 38.0 USDT, OS: 1.2, SS: 1.13, MAD: 2, SOS: 1.6, MSTC: 11


    2021/12 Crash
        ========================================================== BUY TAG STATS ===========================================================
    |   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
    |-------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
    | TOTAL |    412 |           1.14 |         469.43 |          1157.492 |           0.45 |        5:04:00 |   412     0     0   100 |

    2021/05 Crash
        ========================================================== BUY TAG STATS ===========================================================
    |   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
    |-------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
    | TOTAL |    197 |           1.25 |         245.79 |           631.840 |           0.25 |        4:22:00 |   197     0     0   100 |

    2021/09 - 2021/11 Bull
        ========================================================== BUY TAG STATS ===========================================================
    |   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
    |-------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
    | TOTAL |    327 |           1.30 |         424.98 |           961.187 |           0.37 |        3:26:00 |   326     0     1  99.7 |

    """

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.092,
        "29": 0.042,
        "85": 0.03,
        "128": 0.005
    }

    # Sell hyperspace params:
    sell_params = {
        "pHSL": -0.998,

        # 1.1% TTP
        "pPF_1": 0.011,
        "pPF_2": 0.065,
        "pSL_1": 0.011,
        "pSL_2": 0.062,
    }

    # Max Deviation -0.349
    stoploss = -0.998

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # hard stoploss profit
    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Bollinger bands
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=17, stds=1)
        dataframe['bb_lowerband'] = bollinger1['lower']
        dataframe['bb_middleband'] = bollinger1['mid']
        dataframe['bb_upperband'] = bollinger1['upper']

        # Close delta
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # Dip Protection
        dataframe['tpct_change_0']   = top_percent_change(dataframe, 0)
        dataframe['tpct_change_1']   = top_percent_change(dataframe, 1)
        dataframe['tpct_change_2']   = top_percent_change(dataframe, 2)
        dataframe['tpct_change_4']   = top_percent_change(dataframe, 4)
        dataframe['tpct_change_5']   = top_percent_change(dataframe, 5)
        dataframe['tpct_change_9']   = top_percent_change(dataframe, 9)

        # SMA
        dataframe['sma_50'] = ta.SMA(dataframe['close'], timeperiod=50)
        dataframe['sma_200'] = ta.SMA(dataframe['close'], timeperiod=200)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_96'] = williams_r(dataframe, period=96)

        # MAMA / FAMA
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.5, 0.05)
        dataframe['mama_diff'] = ( ( dataframe['mama'] - dataframe['fama'] ) / dataframe['hl2'] )

        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] =  (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        # CTI
        informative['cti_40'] = pta.cti(informative["close"], length=40)

        # %R
        informative['r_96'] = williams_r(informative, period=96)
        informative['r_480'] = williams_r(informative, period=480)

        # 1h mama > fama for general trend check
        informative['hl2'] = (informative['high'] + informative['low']) / 2
        informative['mama'], informative['fama'] = ta.MAMA(informative['hl2'], 0.5, 0.05)
        informative['mama_diff'] = ( ( informative['mama'] - informative['fama'] ) / informative['hl2'] )

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        is_crash_1 = (
                (dataframe['tpct_change_1'] < 0.08) &
                (dataframe['tpct_change_2'] < 0.08) &
                (dataframe['tpct_change_4'] < 0.10)
            )

        dataframe.loc[
            (
                (
                    # Dip check
                    (dataframe['close'] < dataframe['mama']) &
                    (dataframe['r_14'] < -30) &
                    (dataframe['cti'] < 3.0) &
                    (dataframe['adx'] > 26) &
                    (dataframe['mama_diff_1h'] > 0.003) &

                    # Bull confirm
                    (dataframe['mama'] > dataframe['fama']) &
                    (dataframe['sma_50'] > dataframe['sma_200'] * 1.01) &
                    (dataframe['mama_1h'] > dataframe['fama_1h'] * 1.01) &

                    # Overpump check
                    (dataframe['rsi_84'] < 55) &
                    (dataframe['rsi_112'] < 55) &
                    (dataframe['cti_40_1h'] < 0.73) &
                    (dataframe['r_96_1h'] < -6) &
                    (dataframe['mama_diff_1h'] < 0.027) &
                    (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.03 ))
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband'] * 0.999) &
                (dataframe['rsi'] > 76)
            ),
            'sell'] = 0
        return dataframe

class StarRise_dca (StarRise):

    # DCA options
    position_adjustment_enable = True

    initial_safety_order_trigger = -0.016
    max_safety_orders = 11
    safety_order_step_scale = 1.13         #SS
    safety_order_volume_scale = 1.2        #OS

    # Auto compound calculation
    max_dca_multiplier = (1 + max_safety_orders)
    if (max_safety_orders > 0):
        if (safety_order_volume_scale > 1):
            max_dca_multiplier = (2 + (safety_order_volume_scale * (math.pow(safety_order_volume_scale, (max_safety_orders - 1)) - 1) / (safety_order_volume_scale - 1)))
        elif (safety_order_volume_scale < 1):
            max_dca_multiplier = (2 + (safety_order_volume_scale * (1 - math.pow(safety_order_volume_scale, (max_safety_orders - 1))) / (1 - safety_order_volume_scale)))

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:

        if self.config['stake_amount'] == 'unlimited':
            return proposed_stake / self.max_dca_multiplier

        return proposed_stake

    # DCA
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if current_profit > self.initial_safety_order_trigger:
            return None

        count_of_buys = trade.nr_of_successful_buys

        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
            if (self.safety_order_step_scale > 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
            elif (self.safety_order_step_scale < 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale,(count_of_buys - 1))) / (1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    # This calculates base order size
                    stake_amount = stake_amount / self.max_dca_multiplier
                    # This then calculates current safety order size
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None
