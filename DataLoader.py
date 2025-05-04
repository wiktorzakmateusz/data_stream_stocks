import yfinance as yahooFinance
import pandas as pd
import numpy as np

class DataLoader:

    def get_data(self, ticker, time_range):
        """
        time_range values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """    

        # getting stock data
        data = yahooFinance.Ticker(ticker).history(period=time_range)

        # converting datetime index
        data.index.name = 'datetime'
        data.index = data.index.strftime('%Y-%m-%d')
        data.reset_index(inplace=True)
        

        # naming columns
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']

        # saving data (ignoring dividends and stock_splits as they are empty)
        self.raw_data = data.loc[:,['date', 'open', 'close', 'high', 'low', 'volume']]

    def get_data_locally(self, ticker):

        data = pd.read_csv(f'{ticker}.csv')

        # naming columns
        data.columns = ['date', 'close', 'volume', 'open', 'high', 'low']

        for column in ['close', 'open', 'high', 'low']:
            data[column] = data[column].apply(lambda x: float(str(x)[1:]))

        # saving data 
        self.raw_data = data.loc[:,['open', 'close', 'high', 'low', 'volume']]

    def add_features_and_target(self, target_range=1):
        """
        Feature Engineering.
        Creates additional columns based on raw OHLC data. Groups it creates:
        - MIN, MAX close prices for previous sessions
        - SMA (Simple Moving Average) for given number of days
        - OHLC dependance columns
        - Stock Indicators (RSI, MACD, K%D etc.)

        Also it extracts target (always close price) basen on 'target', 'target_range' and 'target_smoothing' parameters.

        -------- parameters --------
        - target (str) - categorial parameter: 
            'changes' - predicting percentage changes in close price
            'normal' - predicting nominal value of close price
        - 'target_range' (int) - sets for how many sessions forward prediction is calculated (usually 1; default=1)
        - 'target_smoothing' (int) - sets for how many sessions an average of target is calculated 
        (if 3 then target is mean of original target price and target price before and after that target,
            if 1 then target is not smoothened - default;
        It sometimes gives better prediction as it lowers fluctuations) 
        """

        # Saving parameters to model
        self.target_range = target_range
        
        # Copying raw data to work on it
        data = self.raw_data.copy()

        ######## features #########
        ###########################

        # max and min values in the preceding sessions for different number of sessions
        for session_range in [5,10,20,40,80,125,250,500,len(data)]: # 
            
            # changing min/max of whole df to 'lifetime'
            if session_range != len(data): session_range_name = session_range 
            else: session_range_name = 'lifetime'

            # initialising columns
            data['max_{}'.format(session_range_name)] = None
            data['min_{}'.format(session_range_name)] = None

            for i in range(1,len(data)):
                starting_point = i - session_range if i-session_range > 0 else 0 # if i-session_range is out of df then 0

                data.loc[i, 'max_{}'.format(session_range_name)] = max(data['close'][starting_point+1:i+1]) # [i-session_range, i] range
                data.loc[i, 'min_{}'.format(session_range_name)] = min(data['close'][starting_point+1:i+1]) # [i-session_range, i] range
        
        # simple moving average values for different number of sessions
        for session_range in [5,10,20,50,100,200]:
            data['sma_{}'.format(session_range)] = data['close'].rolling(session_range).mean()

        # returns in respect of previous sessions
        data['daily_return'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)         # return in % in respect of previous session based on close price
        data['1_week_return'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)        # return in % in respect of previous week session based on close price
        data['2_weeks_return'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)     # return in % in respect of previous 2 weeks session based on close price
        data['1_month_return'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)     # return in % in respect of previous 4 weeks session based on close price
        data['6_months_return'] = (data['close'] - data['close'].shift(120)) / data['close'].shift(120)  # return in % in respect of previous 6 months session based on close price
        data['12_months_return'] = (data['close'] - data['close'].shift(250)) / data['close'].shift(250) # return in % in respect of previous year session based on close price
        
        # dependance between OHLC columns
        data['day_variation'] = (data['high'] - data['low']) / data['low']            # volatility of the price durring the session
        data['day_change'] = (data['close'] - data['open']) / data['open']            # change of the price durring the session
        data['downward_pressure'] = (data['close'] - data['high']) / data['high']     # pressure from sellers, that dropped the price from the highest level
        data['upward_pressure'] = (data['close'] - data['low']) / data['low']         # pressure from buyers, that lifted the price from the lowest level


        # Stock indicators

        # Relative Strength index
        def rsi_calculation(data, session_range=14):
            daily_return = data['close'] - data['close'].shift(1)

            daily_return_positive = daily_return.copy()
            daily_return_negative = daily_return.copy()

            daily_return_positive[daily_return_positive<0] = 0
            daily_return_negative[daily_return_negative>0] = 0

            daily_return_positive_average = daily_return_positive.rolling(session_range).mean().abs()
            daily_return_negative_average = daily_return_negative.rolling(session_range).mean().abs()

            rsi = 100 * daily_return_positive_average / (daily_return_positive_average + daily_return_negative_average) # other evaluation 100 - 100 / (1 + (daily_return_positive_average / daily_return_negative_average))

            rsi[daily_return_positive_average == 0] = 0 
            rsi[daily_return_negative_average == 0] = 100 

            return rsi

        # Moving Average Convergence Divergence
        def macd_calculation(data, short_period=12, long_period=26):

            short_ema = data['close'].ewm(span=short_period, adjust=False).mean()
            long_ema = data['close'].ewm(span=long_period, adjust=False).mean()

            macd = short_ema - long_ema

            return macd

        # Percentage Price Oscillator 
        def percentage_price_oscillator(data, short_period=12, long_period=26):
            ema_short = data['close'].ewm(span=short_period, adjust=False).mean()
            ema_long = data['close'].ewm(span=long_period, adjust=False).mean()
            ppo = (ema_short - ema_long) / ema_long * 100
            return ppo

        # Stochastic Oscillator (K%D)
        def stochastic_calculation(data, session_range=14):

            low_min = data['low'].rolling(session_range).min()
            high_max = data['high'].rolling(session_range).max()

            stochastic_fast = (data['close'] - low_min) / (high_max - low_min) * 100 # fast K
            stochastic_slow = stochastic_fast.rolling(3).mean()                      # slow D                     

            return stochastic_fast, stochastic_slow

        # %R Williams Oscillator
        def williams_r(data, period=10):
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            wr = 100 * ((high_max - data['close']) / (high_max - low_min))
            return wr

        # Average True Range
        def calculate_atr(data, window=14):
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            return atr

        # Chande Momentum Oscillator
        def chande_momentum_oscillator(data, period=10):
            diff = data['close'].diff()
            gain = diff.where(diff > 0, 0).rolling(window=period).sum()
            loss = -diff.where(diff < 0, 0).rolling(window=period).sum()
            cmo = 100 * (gain - loss) / (gain + loss)
            cmo[cmo.apply(lambda x: pd.isna(x))] = 0
            return cmo

        # Commodity Channel Index
        def commodity_channel_index(data, period=20):
            tp = (data['high'] + data['low'] + data['close']) / 3
            ma = tp.rolling(window=period).mean()
            mad = (tp - ma).abs().rolling(window=period).mean()
            cci = (tp - ma) / (0.015 * mad)
            return cci

        # Momentum (MOM)
        def momentum(data, period=10):
            mom = data['close'] - data['close'].shift(period)
            return mom

        # BIAS (Bias Indicator)
        def bias(data, period=10):
            ma = data['close'].rolling(window=period).mean()
            bias = (data['close'] - ma) / ma * 100
            return bias

        # Weighted Negative Ratio
        def weighted_negative_ratio(data, period=10):
            diff = data['close'].diff()
            wn_ratio = diff.where(diff < 0, 0).rolling(window=period).sum() / diff.abs().rolling(window=period).sum()
            wn_ratio[wn_ratio.apply(lambda x: pd.isna(x))] = -0.5

            return wn_ratio

        # calculating indicators
        data['rsi'] = rsi_calculation(data) 
        data['macd'] = macd_calculation(data) 
        data['ppo'] = percentage_price_oscillator(data)
        data['stochastic_fast'], data['stochastic_slow'] = stochastic_calculation(data) 
        data[r'%r'] = williams_r(data)
        data['atr'] = calculate_atr(data)
        data['cmo'] = chande_momentum_oscillator(data)
        data['cci'] = commodity_channel_index(data)
        data['mom'] = momentum(data)
        data['bias'] = bias(data)
        data['wnr'] = weighted_negative_ratio(data)
        

        ######## target #########
        #########################

        # setting target based on 'target_range' (range of prediction)
        data['target'] = (data['close'].shift(-self.target_range) - data['close']).apply(lambda x: 1 if x > 0 else -1)
        
        # dropping N/A records that occured due to feature engineering
        data = data.iloc[250: -self.target_range, :]

        # checking if data does not contain N/A records any more
        if data.isnull().any().any(): 
            print('There is N/A value in the dataframe! Check source of data!')

        # reseting index of df
        data.reset_index(drop=True, inplace=True)

        # storing data to class variable
        self.data_with_target = data

    def pipeline(self, ticker):
        self.get_data('AAPL', 'max')
        self.raw_data.drop('date', axis=1, inplace=True)
        self.add_features_and_target()
        
        return self.data_with_target
